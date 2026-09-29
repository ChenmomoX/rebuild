"""Build physiobank_clean.npy / physiobank_contaminated.npy for data.py's 'phy' branch.

Source: PhysioNet "Motion Artifact Contaminated fNIRS and EEG Data" v1.0.0
(Sweeney et al., IEEE TBiT 2012; DOI 10.13026/C2988P). Each of the 23 EEG trials
has two closely spaced pre-frontal EEG channels sampled at 2048 Hz: one transducer
was left undisturbed (ground truth) while the other was manipulated to induce
motion artifacts.

Pipeline (documented reconstruction; the KBS paper authors' exact preprocessing
is not public):
  1. trim each trial to the EEG-trigger high interval (start/end of experiment)
  2. anti-alias resample both EEG channels 2048 Hz -> 256 Hz (resample_poly, 8x)
  3. per trial, the channel with the larger robust scale (MAD-based) is taken as
     the motion-contaminated channel, the other one as the clean channel
  4. cut both channels into non-overlapping 1 s (256-sample) epochs, removing
     each epoch's own DC offset
     -> physiobank_clean.npy, physiobank_contaminated.npy, shape (N, 256)

The per-epoch DC removal in step 4 is required: the two transducers sit at
different electrode DC potentials, and data.py later normalizes the clean
channel with the noisy channel's per-epoch mean/std — without it the clean
targets end up offset by hundreds of std units and the SNR/RRMSE metrics
become meaningless.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

RAW_DIR = "physionet_raw"
SFREQ_OUT = 256
DECIM = 8  # 2048 / 256

clean_epochs, contaminated_epochs = [], []

for path in sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)):
    df = pd.read_csv(path, header=None)
    ch1 = df.iloc[:, 1].to_numpy(dtype=np.float64)
    ch2 = df.iloc[:, 2].to_numpy(dtype=np.float64)
    trig = df.iloc[:, 3].to_numpy(dtype=np.float64)

    # trim to experiment interval: trigger high level = experiment running
    hi = np.median(trig[trig > 0]) if (trig > 0).any() else None
    if hi is not None:
        active = np.abs(trig - hi) < 0.5 * hi
        if active.any():
            first, last = np.argmax(active), len(active) - np.argmax(active[::-1])
            ch1, ch2 = ch1[first:last], ch2[first:last]

    n = min(len(ch1), len(ch2)) // DECIM * DECIM
    ch1 = resample_poly(ch1[:n], 1, DECIM)
    ch2 = resample_poly(ch2[:n], 1, DECIM)

    # motion artifacts are episodic: the contaminated channel shows a much higher
    # spread (p90/p50) of per-second std than the undisturbed ground-truth channel
    def episodicity(x):
        s = np.std(x[: len(x) // SFREQ_OUT * SFREQ_OUT].reshape(-1, SFREQ_OUT), axis=1)
        return float(np.percentile(s, 90) / np.median(s))

    dirty_ch = 1 if episodicity(ch1) > episodicity(ch2) else 2
    if dirty_ch == 1:
        dirty, clean = ch1, ch2
    else:
        dirty, clean = ch2, ch1

    n_seg = min(len(clean), len(dirty)) // SFREQ_OUT

    def epochs_zero_mean(x):
        ep = x[: n_seg * SFREQ_OUT].reshape(n_seg, SFREQ_OUT)
        return ep - ep.mean(axis=1, keepdims=True)

    clean_epochs.append(epochs_zero_mean(clean))
    contaminated_epochs.append(epochs_zero_mean(dirty))
    print(f"{os.path.basename(path)}: {n_seg} epochs, contaminated = channel {dirty_ch} "
          f"(episodicity {max(episodicity(ch1), episodicity(ch2)):.2f} vs "
          f"{min(episodicity(ch1), episodicity(ch2)):.2f})")

clean_all = np.concatenate(clean_epochs, axis=0)[:, None, :].astype(np.float32)
dirty_all = np.concatenate(contaminated_epochs, axis=0)[:, None, :].astype(np.float32)
np.save("physiobank_clean.npy", clean_all)
np.save("physiobank_contaminated.npy", dirty_all)
print("saved:", clean_all.shape, dirty_all.shape)
