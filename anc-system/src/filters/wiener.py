"""
Single-channel spectral-subtraction / Wiener-gain baseline. No reference
channel needed — the noise spectrum is estimated from the first
`noise_estimate_ms` milliseconds of each clip (or a supplied silence-only
snippet). This is the classical "before AI" baseline mentioned in the plan
(Phase 2), and the cheap first pass the DL model refines in the hybrid path.
"""
import numpy as np


def wiener_denoise(signal: np.ndarray, sr: int, noise_estimate_ms: float = 300,
                    n_fft: int = 512, hop_length: int = 128, floor: float = 0.05):
    noise_samples = int(sr * noise_estimate_ms / 1000)
    noise_clip = signal[:max(noise_samples, n_fft)]

    stft = librosa_stft(signal, n_fft, hop_length)
    noise_stft = librosa_stft(noise_clip, n_fft, hop_length)

    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
    sig_power = np.abs(stft) ** 2

    gain = (sig_power - noise_power) / (sig_power + 1e-12)
    gain = np.clip(gain, floor, 1.0)  # floor avoids musical-noise artifacts

    denoised_stft = stft * gain
    return librosa_istft(denoised_stft, hop_length, length=len(signal))


def librosa_stft(x, n_fft, hop_length):
    import librosa
    return librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop_length)


def librosa_istft(stft, hop_length, length):
    import librosa
    return librosa.istft(stft, hop_length=hop_length, length=length)
