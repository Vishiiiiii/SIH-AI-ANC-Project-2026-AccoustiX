"""
Single-channel Wiener-gain filter using minimum-statistics noise tracking
(Martin, 1994-style): a running minimum of the smoothed power spectrum
serves as a continuously-updated noise floor estimate, instead of assuming
a fixed noise-only lead-in segment.

Why: our synthetic mixtures have noise present across the ENTIRE clip (see
mix.py), so there's no clean "noise-only" snippet to estimate from up
front. Minimum-statistics tracking sidesteps this: speech naturally dips
in level between words/phrases, and during those dips the smoothed power
briefly approaches the true noise floor. Tracking a leaky running minimum
recovers that floor over time, continuously, causally, with no lookahead
beyond the current STFT frame — safe for real-time use.
"""
import numpy as np


class WienerFilter:
    def __init__(self, n_fft: int = 512, hop_length: int = 128, floor: float = 0.05,
                 alpha_smooth: float = 0.8, min_rise: float = 1.005, bias_correction: float = 1.5):
        """
        alpha_smooth: EMA smoothing on the power spectrum (higher = smoother/slower).
        min_rise: how fast the running minimum is allowed to creep back up
                  between dips (>1.0; closer to 1.0 = slower rise = more stable
                  but slower to adapt to noise that's genuinely getting louder).
        bias_correction: minimum-tracking systematically underestimates true
                  noise power (it's a minimum, after all) — this compensates.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.floor = floor
        self.alpha_smooth = alpha_smooth
        self.min_rise = min_rise
        self.bias_correction = bias_correction
        self.smoothed_power = None   # (freq_bins,) — lazy-init on first frame
        self.min_power = None

    def reset(self):
        """Call this between unrelated clips/streams so noise history doesn't leak across them."""
        self.smoothed_power = None
        self.min_power = None

    def process(self, signal: np.ndarray) -> np.ndarray:
        stft = self._stft(signal)                  # (freq_bins, frames)
        power = np.abs(stft) ** 2

        freq_bins, n_frames = power.shape
        if self.smoothed_power is None:
            self.smoothed_power = power[:, 0].copy()
            self.min_power = power[:, 0].copy()

        gain = np.zeros_like(power)
        for t in range(n_frames):
            self.smoothed_power = (self.alpha_smooth * self.smoothed_power
                                    + (1 - self.alpha_smooth) * power[:, t])
            # Leaky minimum: instantly track downward, creep upward slowly.
            self.min_power = np.minimum(self.smoothed_power, self.min_power * self.min_rise)

            noise_power = self.min_power * self.bias_correction
            g = (power[:, t] - noise_power) / (power[:, t] + 1e-12)
            gain[:, t] = np.clip(g, self.floor, 1.0)

        denoised_stft = stft * gain
        return self._istft(denoised_stft, length=len(signal))

    def _stft(self, x):
        import librosa
        return librosa.stft(x.astype(np.float32), n_fft=self.n_fft, hop_length=self.hop_length)

    def _istft(self, stft, length):
        import librosa
        return librosa.istft(stft, hop_length=self.hop_length, length=length)


def wiener_denoise(signal: np.ndarray, sr: int, **kwargs):
    """One-shot convenience wrapper for offline use (e.g. precomputing a
    training set) where you don't need state to persist across calls."""
    return WienerFilter(**kwargs).process(signal)
