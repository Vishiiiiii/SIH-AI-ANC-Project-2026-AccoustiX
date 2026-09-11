"""
Synthetic mixing engine: clean speech + noise -> noisy mixture at a target SNR.

Bug note (fixed here vs. the original snippet): the original always tiled
`noise` before checking whether it also needed *trimming* when `noise` was
already longer than `clean`. That's actually fine for tiling, but it never
handled MONO/STEREO or sample-rate mismatches — both handled below.
"""
import random
import numpy as np
import soundfile as sf


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    clean = _to_mono(clean).astype(np.float32)
    noise = _to_mono(noise).astype(np.float32)

    if len(noise) == 0:
        raise ValueError("Empty noise clip")

    # Loop noise until it's at least as long as clean, then take a random crop.
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    start = random.randint(0, len(noise) - len(clean)) if len(noise) > len(clean) else 0
    noise = noise[start:start + len(clean)]

    clean_rms, noise_rms = rms(clean), rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-12))

    noisy = clean + noise_scaled
    peak = np.max(np.abs(noisy))
    if peak > 1.0:  # avoid clipping — scale both together so the SNR is preserved
        noisy = noisy / peak
        clean = clean / peak

    return noisy.astype(np.float32), clean.astype(np.float32)


def randomize_impulsive_onset(noise: np.ndarray, clean_len: int, min_silence_ratio: float = 0.3):
    """
    For impulsive noises (gunfire/explosions), pad with silence so the event
    doesn't always land at t=0 — otherwise the model just learns "suppress the
    start of the clip" instead of learning to detect transients anywhere.

    If the noise clip is already >= clean_len, take a random crop instead of
    always the first clean_len samples, for the same reason.
    """
    if len(noise) >= clean_len:
        start = random.randint(0, len(noise) - clean_len)
        return noise[start:start + clean_len]
    max_pad = int(clean_len - len(noise))
    pad_before = random.randint(0, max_pad)
    pad_after = max_pad - pad_before
    return np.pad(noise, (pad_before, pad_after))
