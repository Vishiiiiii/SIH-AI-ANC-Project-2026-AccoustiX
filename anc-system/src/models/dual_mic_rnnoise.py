"""A low-latency RNNoise-style mask estimator that uses both microphones.

Input channel 0 is the NLMS/Wiener residual (the signal sent to the neural
stage).  Input channel 1 is NLMS's aligned estimate of the primary-mic noise.
This is deliberately more useful than feeding the raw reference mic: NLMS has
already learnt its gain and delay relative to the primary microphone.
"""
import torch
import torch.nn as nn

from src.models.stft_utils import ConvSTFT, ConvISTFT


class DualMicRNNoiseStyle(nn.Module):
    input_channels = 2
    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 gru_hidden: int = 256, conv_channels: int = 256):
        super().__init__()
        self.stft = ConvSTFT(n_fft, hop_length)
        self.istft = ConvISTFT(n_fft, hop_length)
        freq_bins = n_fft // 2 + 1
        # Residual magnitude, aligned-noise magnitude, and their signed
        # spectral similarity let the mask estimator exploit the second mic.
        self.conv = nn.Sequential(
            nn.Conv1d(freq_bins * 3, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.PReLU(),
            nn.Conv1d(conv_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.PReLU(),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden, batch_first=True)
        self.mask_head = nn.Sequential(nn.Linear(gru_hidden, freq_bins), nn.Sigmoid())

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 3 or waveforms.shape[1] != 2:
            raise ValueError("DualMicRNNoiseStyle expects [batch, 2, samples].")
        residual, noise_estimate = waveforms[:, 0], waveforms[:, 1]
        real, imag = self.stft(residual)
        noise_real, noise_imag = self.stft(noise_estimate)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        noise_magnitude = torch.sqrt(noise_real.square() + noise_imag.square() + 1e-8)
        similarity = (real * noise_real + imag * noise_imag) / (magnitude * noise_magnitude + 1e-8)
        features = torch.cat((torch.log1p(magnitude), torch.log1p(noise_magnitude), similarity), dim=1)
        features = self.conv(features).transpose(1, 2)
        features, _ = self.gru(features)
        # A [0, 2] real-valued ratio mask starts at unity but can restore
        # speech that the conservative Wiener stage attenuated.  A [0, 1]
        # mask can only remove energy, which caps attainable PESQ here.
        mask = 2.0 * self.mask_head(features).transpose(1, 2)
        return self.istft(real * mask, imag * mask, length=residual.shape[-1])
