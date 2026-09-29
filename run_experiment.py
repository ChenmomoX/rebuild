"""Reproducible training entry point for Self-Supervised-EEG-Denoising.

Wraps the original train.py logic (ModelTrainer / NoisyDataset1D / DenoiseEEG)
with what it was missing:
  - argparse CLI and fixed random seed (mask and init are the only stochastic parts)
  - per-epoch CSV metrics log
  - best-by-val-SNR checkpoint saving (the original code only printed metrics)
  - export of denoised validation samples for plotting

Examples:
  python run_experiment.py --dataset semi --epochs 2          # smoke test
  python run_experiment.py --dataset phy --epochs 100         # full run
  python plot_results.py results/phy_e100_seed42_20260929_160000
"""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import data
from model import DenoiseEEG
from train import ModelTrainer, NoisyDataset1D, TrainingConfig

SNR_LEVEL = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
METRIC_NAMES = ['val_loss', 'rrmse_t', 'rrmse_s', 'cc', 'snr_db']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='phy',
                   choices=['phy', 'EMG', 'EOG', 'hybrid', 'semi'],
                   help="must match the .npy files present in the working directory")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--mask-ratio', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-samples', type=int, default=24,
                   help='number of validation samples exported to denoised_samples.npz')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--outdir', default='results')
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloaders(dataset_type, batch_size, num_workers):
    # same 80/20 split as train.prepare_data (random_state=42), but with a
    # configurable batch size
    clean_all, noisy_all = data.getdata(dataset_type, SNR_LEVEL)
    clean_train, clean_val, noisy_train, noisy_val = train_test_split(
        clean_all, noisy_all, test_size=0.2, random_state=42, shuffle=True)
    train_loader = DataLoader(NoisyDataset1D(clean_train, noisy_train),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(NoisyDataset1D(clean_val, noisy_val),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


def build_model(dataset_type, device):
    cfg = TrainingConfig().dataset_configs[dataset_type]
    return DenoiseEEG(cfg['input_channels'], cfg['seq_len'], cfg['hidden_dim']).to(device)


def export_samples(model, val_loader, device, n_samples, run_dir, args):
    model.eval()
    noisy_l, clean_l, den_l = [], [], []
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x.to(device)).cpu()
            noisy_l.append(x)
            clean_l.append(y)
            den_l.append(out)
            n += x.shape[0]
            if n >= n_samples:
                break
    np.savez_compressed(
        os.path.join(run_dir, 'denoised_samples.npz'),
        noisy=torch.cat(noisy_l)[:n_samples].numpy(),
        clean=torch.cat(clean_l)[:n_samples].numpy(),
        denoised=torch.cat(den_l)[:n_samples].numpy(),
        dataset=args.dataset,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    run_name = (f"{args.dataset}_e{args.epochs}_seed{args.seed}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir = os.path.join(args.outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_dump = dict(vars(args))
    cfg_dump.update({
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'device': str(device),
        'cuda_device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    })
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg_dump, f, indent=2, ensure_ascii=False)

    print(f'run dir: {run_dir}', flush=True)
    print(f'loading dataset "{args.dataset}" ...', flush=True)
    train_loader, val_loader = make_dataloaders(args.dataset, args.batch_size,
                                                args.num_workers)
    print(f'train batches: {len(train_loader)}, val batches: {len(val_loader)}, '
          f'device: {device}', flush=True)

    model = build_model(args.dataset, device)
    trainer = ModelTrainer(TrainingConfig())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    csv_path = os.path.join(run_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', *METRIC_NAMES, 'elapsed_s', 'is_best'])

    best = {'snr_db': -1e9, 'epoch': -1, 'state': None, 'metrics': None}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(model, train_loader, optimizer,
                                         args.mask_ratio)
        val_loss, cc, snr_db, rrmse_t, rrmse_s = trainer.evaluate(model, val_loader)
        vals = {name: float(v) for name, v in
                zip(METRIC_NAMES, [val_loss, rrmse_t, rrmse_s, cc, snr_db])}
        elapsed = time.time() - t0

        is_best = vals['snr_db'] > best['snr_db']
        if is_best:
            best = {'snr_db': vals['snr_db'], 'epoch': epoch,
                    'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'metrics': vals}
            torch.save({'epoch': epoch, 'args': vars(args),
                        'metrics': vals, 'model_state': best['state']},
                       os.path.join(run_dir, 'checkpoint_best.pt'))

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                [epoch, f'{train_loss:.6f}', *[f'{vals[m]:.6f}' for m in METRIC_NAMES],
                 f'{elapsed:.1f}', is_best])

        eta_min = (time.time() - t_start) / epoch * (args.epochs - epoch) / 60
        print(f'[{run_name}] epoch {epoch}/{args.epochs}, '
              f"train_loss {train_loss:.4f}, val_loss {vals['val_loss']:.4f}, "
              f"SNR {vals['snr_db']:.3f}, CC {vals['cc']:.3f}, "
              f"RRMSE_t {vals['rrmse_t']:.3f}, {elapsed:.1f}s/epoch, "
              f'ETA {eta_min:.1f} min' + (' *best*' if is_best else ''),
              flush=True)

    torch.save({'epoch': args.epochs, 'args': vars(args),
                'model_state': model.state_dict()},
               os.path.join(run_dir, 'checkpoint_last.pt'))

    model.load_state_dict(best['state'])
    export_samples(model, val_loader, device, args.save_samples, run_dir, args)

    m = best['metrics']
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min. "
          f"best epoch {best['epoch']}: SNR {m['snr_db']:.3f}, CC {m['cc']:.3f}, "
          f"RRMSE_t {m['rrmse_t']:.3f}, RRMSE_s {m['rrmse_s']:.3f}",
          flush=True)


if __name__ == '__main__':
    sys.exit(main())
