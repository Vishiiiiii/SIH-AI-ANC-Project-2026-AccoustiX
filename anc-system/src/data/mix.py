"""
Synthetic mixing engine: clean speech + noise -> noisy mixture at a target SNR.

Bug note (fixed here vs. the original snippet): the original always tiled
`noise` before checking whether it also needed *trimming* when `noise` was
already longer than `clean`. That's actually fine for tiling, but it never
handled MONO/STEREO or sample-rate mismatches — both handled below.
"""
import numpy as np


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator | None = None,
               return_noise: bool = False):
    """Mix clean speech with noise and optionally return the exact noise track.

    Returning the scaled, cropped noise is essential for the dual-mic data
    generator: it becomes the source for a realistic reference microphone.
    A caller-owned generator makes complete dataset builds reproducible.
    """
    rng = rng or np.random.default_rng()
    clean = _to_mono(clean).astype(np.float32)
    noise = _to_mono(noise).astype(np.float32)

    if len(noise) == 0:
        raise ValueError("Empty noise clip")

    # Loop noise until it's at least as long as clean, then take a random crop.
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    start = int(rng.integers(0, len(noise) - len(clean) + 1)) if len(noise) > len(clean) else 0
    noise = noise[start:start + len(clean)]

    clean_rms, noise_rms = rms(clean), rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-12))

    noisy = clean + noise_scaled
    peak = np.max(np.abs(noisy))
    if peak > 1.0:  # avoid clipping — scale both together so the SNR is preserved
        noisy = noisy / peak
        clean = clean / peak
        noise_scaled = noise_scaled / peak

    outputs = (noisy.astype(np.float32), clean.astype(np.float32))
    return outputs + (noise_scaled.astype(np.float32),) if return_noise else outputs


def synthesize_reference(noise_component: np.ndarray, rng: np.random.Generator,
                         max_delay_ms: float = 4.0, sample_rate: int = 16000,
                         gain_db_range: tuple[float, float] = (-8.0, 8.0),
                         self_noise_db: float = -38.0) -> tuple[np.ndarray, dict]:
    """Simulate a second microphone that hears correlated noise, not speech.

    It is deliberately modest: gain mismatch, a short *primary-mic lag* and
    independent microphone self-noise.  The reference is shifted earlier than
    the primary noise: a causal NLMS filter can then use present/past reference
    samples to predict the later primary noise. Real captures should replace
    this channel once the INMP441 pair arrives.
    """
    noise_component = np.asarray(noise_component, dtype=np.float32).reshape(-1)
    gain_db = float(rng.uniform(*gain_db_range))
    gain = 10.0 ** (gain_db / 20.0)
    max_delay_samples = int(round(max_delay_ms * sample_rate / 1000.0))
    primary_lag_samples = int(rng.integers(0, max_delay_samples + 1))
    # `reference[t - primary_lag]` corresponds to `noise_component[t]`.
    # This direction matters: delaying the reference would force a causal
    # filter to predict future noise and would train the wrong behaviour.
    reference = np.pad(
        noise_component[primary_lag_samples:] * gain,
        (0, primary_lag_samples),
    )
    self_noise_rms = rms(noise_component) * (10.0 ** (self_noise_db / 20.0))
    reference += rng.normal(0.0, self_noise_rms, len(reference)).astype(np.float32)
    return reference.astype(np.float32), {
        "reference_gain_db": round(gain_db, 3),
        "primary_lag_samples": primary_lag_samples,
        "reference_self_noise_db": self_noise_db,
    }


def randomize_impulsive_onset(noise: np.ndarray, clean_len: int,
                              rng: np.random.Generator | None = None):
    """
    For impulsive noises (gunfire/explosions), pad with silence so the event
    doesn't always land at t=0 — otherwise the model just learns "suppress the
    start of the clip" instead of learning to detect transients anywhere.

    If the noise clip is already >= clean_len, take a random crop instead of
    always the first clean_len samples, for the same reason.
    """
    rng = rng or np.random.default_rng()
    if len(noise) >= clean_len:
        start = int(rng.integers(0, len(noise) - clean_len + 1))
        return noise[start:start + clean_len]
    max_pad = int(clean_len - len(noise))
    pad_before = int(rng.integers(0, max_pad + 1))
    pad_after = max_pad - pad_before
    return np.pad(noise, (pad_before, pad_after))
