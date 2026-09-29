"""Plot training curves and denoising results from a run directory produced by
run_experiment.py (reads metrics.csv and denoised_samples.npz; needs matplotlib
only, torch not required).

Usage:
  python plot_results.py results/phy_e100_seed42_20260929_160000
  python plot_results.py results/semi_e2_seed42_20260929_155000 --num-samples 6
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_FS = {'EMG': 512, 'EOG': 512, 'hybrid': 512, 'semi': 200, 'phy': 256}
EPS = 1e-12


def snr_db(clean, other):
    return 10 * np.log10((clean ** 2).sum(-1) /
                         (((clean - other) ** 2).sum(-1) + EPS))


def read_metrics(csv_path):
    cols = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                if key == 'is_best':
                    cols.setdefault(key, []).append(val == 'True')
                else:
                    cols.setdefault(key, []).append(float(val))
    return {k: np.array(v) for k, v in cols.items()}


def plot_curves(run_dir, m):
    best_rows = np.flatnonzero(m['is_best'])
    best_i = best_rows[-1] if len(best_rows) else int(np.argmax(m['snr_db']))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(m['epoch'], m['train_loss'], label='train loss')
    ax.plot(m['epoch'], m['val_loss'], label='val loss')
    ax.set_title('Loss'); ax.set_xlabel('epoch'); ax.legend()

    ax = axes[0, 1]
    ax.plot(m['epoch'], m['snr_db'])
    ax.set_title('Val SNR (dB)'); ax.set_xlabel('epoch')

    ax = axes[1, 0]
    ax.plot(m['epoch'], m['cc'])
    ax.set_title('Val CC (Pearson)'); ax.set_xlabel('epoch')

    ax = axes[1, 1]
    ax.plot(m['epoch'], m['rrmse_t'], label='RRMSE_t')
    ax.plot(m['epoch'], m['rrmse_s'], label='RRMSE_s')
    ax.set_title('Val RRMSE'); ax.set_xlabel('epoch'); ax.legend()

    for ax in axes.flat:
        ax.axvline(m['epoch'][best_i], color='r', ls='--', lw=1, alpha=0.6)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{os.path.basename(run_dir)}  (best SNR epoch {int(m['epoch'][best_i])})")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'curves.png'), dpi=150)
    plt.close(fig)
    return best_i


def plot_time_domain(run_dir, z, n_samples):
    noisy, clean, den = z['noisy'], z['clean'], z['denoised']
    n_samples = min(n_samples, noisy.shape[0])
    n_ch = noisy.shape[1]
    ch = n_ch // 2 if n_ch > 1 else 0  # for multi-channel, show the middle one
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), sharex=False)
    for k, ax in enumerate(axes.flat):
        s_before = snr_db(clean[k, ch], noisy[k, ch])
        s_after = snr_db(clean[k, ch], den[k, ch])
        ax.plot(noisy[k, ch], color='0.75', lw=0.5, label='noisy')
        ax.plot(clean[k, ch], color='k', lw=0.9, label='clean')
        ax.plot(den[k, ch], color='tab:red', lw=0.7, alpha=0.85, label='denoised')
        ax.set_title(f'#{k}  SNR {s_before:.1f} -> {s_after:.1f} dB', fontsize=10)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{os.path.basename(run_dir)}  time domain (channel {ch})")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'time_domain.png'), dpi=150)
    plt.close(fig)


def plot_psd(run_dir, z, n_samples):
    noisy, clean, den = z['noisy'], z['clean'], z['denoised']
    fs = DATASET_FS[str(z['dataset'])]
    n_ch = noisy.shape[1]
    ch = n_ch // 2 if n_ch > 1 else 0
    n_samples = min(n_samples, noisy.shape[0])
    freqs = np.fft.rfftfreq(noisy.shape[-1], d=1.0 / fs)
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), sharex=True)
    for k, ax in enumerate(axes.flat):
        for sig, color, label in ((noisy, '0.6', 'noisy'),
                                  (clean, 'k', 'clean'),
                                  (den, 'tab:red', 'denoised')):
            spec = np.abs(np.fft.rfft(sig[k, ch]))
            ax.plot(freqs, 20 * np.log10(spec + EPS), color=color, lw=0.7,
                    alpha=0.9, label=label)
        ax.set_title(f'#{k}', fontsize=10)
        ax.set_xlim(0, fs / 2)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    for ax in axes[2]:
        ax.set_xlabel('frequency (Hz)')
    fig.suptitle(f"{os.path.basename(run_dir)}  amplitude spectrum, "
                 f"fs={fs} Hz (channel {ch})")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'psd.png'), dpi=150)
    plt.close(fig)


def write_summary(run_dir, m, best_i):
    path = os.path.join(run_dir, 'best_metrics.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"run: {os.path.basename(run_dir)}\n")
        f.write(f"best epoch (by val SNR): {int(m['epoch'][best_i])} of "
                f"{int(m['epoch'][-1])}\n")
        for key in ('train_loss', 'val_loss', 'rrmse_t', 'rrmse_s', 'cc', 'snr_db'):
            f.write(f"{key}: {m[key][best_i]:.4f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', help='directory produced by run_experiment.py')
    ap.add_argument('--num-samples', type=int, default=9,
                    help='panels per figure (time domain / PSD), max 9')
    args = ap.parse_args()

    m = read_metrics(os.path.join(args.run_dir, 'metrics.csv'))
    best_i = plot_curves(args.run_dir, m)
    write_summary(args.run_dir, m, best_i)

    npz_path = os.path.join(args.run_dir, 'denoised_samples.npz')
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        plot_time_domain(args.run_dir, z, args.num_samples)
        plot_psd(args.run_dir, z, args.num_samples)
    else:
        print('denoised_samples.npz not found; skipped signal/PSD plots')

    print('plots written to', args.run_dir)


if __name__ == '__main__':
    main()
