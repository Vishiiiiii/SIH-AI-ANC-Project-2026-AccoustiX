"""Fast Wiener spectral denoiser used as an optional classical stage.

This deliberately uses SciPy rather than librosa.  Librosa lazily imports
Numba on the first callback, which is both too slow for live audio and fails
in some Python 3.13 environments.  With two microphones, pass the reference
channel so the noise estimate is based on actual noise rather than assuming
the beginning of each block is silence.
"""
import numpy as np
from scipy import signal as scipy_signal


class WienerFilter:
    def __init__(self, noise_estimate_ms: float = 50, n_fft: int = 512,
                 hop_length: int = 128, floor: float = 0.05):
        self.noise_estimate_ms = noise_estimate_ms
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.floor = floor

    def process(self, waveform: np.ndarray, sr: int,
                noise_reference: np.ndarray | None = None) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not len(waveform):
            return waveform.copy()
        if noise_reference is None:
            noise_samples = max(int(sr * self.noise_estimate_ms / 1000), self.n_fft)
            noise_clip = waveform[:noise_samples]
        else:
            noise_clip = np.asarray(noise_reference, dtype=np.float32).reshape(-1)
            if not len(noise_clip):
                raise ValueError("noise_reference must not be empty")

        noverlap = self.n_fft - self.hop_length
        _, _, stft = scipy_signal.stft(
            waveform, fs=sr, window="hann", nperseg=self.n_fft,
            noverlap=noverlap, boundary="zeros", padded=True,
        )
        _, _, noise_stft = scipy_signal.stft(
            noise_clip, fs=sr, window="hann", nperseg=self.n_fft,
            noverlap=noverlap, boundary="zeros", padded=True,
        )
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
        signal_power = np.abs(stft) ** 2
        gain = np.clip(
            (signal_power - noise_power) / (signal_power + 1e-12),
            self.floor,
            1.0,
        )
        _, enhanced = scipy_signal.istft(
            stft * gain, fs=sr, window="hann", nperseg=self.n_fft,
            noverlap=noverlap, input_onesided=True, boundary=True,
        )
        if len(enhanced) < len(waveform):
            enhanced = np.pad(enhanced, (0, len(waveform) - len(enhanced)))
        return enhanced[:len(waveform)].astype(np.float32)


def wiener_denoise(waveform: np.ndarray, sr: int, noise_estimate_ms: float = 50,
                    n_fft: int = 512, hop_length: int = 128, floor: float = 0.05,
                    noise_reference: np.ndarray | None = None) -> np.ndarray:
    """Stateless convenience wrapper for offline processing and callbacks."""
    return WienerFilter(noise_estimate_ms, n_fft, hop_length, floor).process(
        waveform, sr, noise_reference=noise_reference,
    )
