"""Three-input, complex-mask dual-microphone speech enhancer.

The network gets the live front-end residual, NLMS's primary-aligned noise
estimate, and the original primary microphone.  The last channel preserves
speech detail that the cautious classical Wiener stage may have attenuated.
It predicts a complex mask, so it can correct phase as well as magnitude.
"""
import torch
import torch.nn as nn

from src.models.stft_utils import ConvSTFT, ConvISTFT


class DualMicComplexRNNoiseStyle(nn.Module):
    input_channels = 3

    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 gru_hidden: int = 192, conv_channels: int = 256):
        super().__init__()
        self.stft = ConvSTFT(n_fft, hop_length)
        self.istft = ConvISTFT(n_fft, hop_length)
        freq_bins = n_fft // 2 + 1
        self.conv = nn.Sequential(
            nn.Conv1d(freq_bins * 4, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.PReLU(),
            nn.Conv1d(conv_channels, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.PReLU(),
        )
        # Within the fixed 128 ms block, backward context does not add any
        # buffering delay. It helps recover consonants around impulsive noise.
        self.gru = nn.GRU(192, gru_hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.mask_head = nn.Linear(gru_hidden * 2, freq_bins * 2)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 3 or waveforms.shape[1] != self.input_channels:
            raise ValueError("DualMicComplexRNNoiseStyle expects [batch, 3, samples].")
        residual, noise_estimate, primary = waveforms[:, 0], waveforms[:, 1], waveforms[:, 2]
        real, imag = self.stft(residual)
        noise_real, noise_imag = self.stft(noise_estimate)
        primary_real, primary_imag = self.stft(primary)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        noise_magnitude = torch.sqrt(noise_real.square() + noise_imag.square() + 1e-8)
        primary_magnitude = torch.sqrt(primary_real.square() + primary_imag.square() + 1e-8)
        similarity = (real * noise_real + imag * noise_imag) / (magnitude * noise_magnitude + 1e-8)
        features = torch.cat((torch.log1p(magnitude), torch.log1p(noise_magnitude),
                              torch.log1p(primary_magnitude), similarity), dim=1)
        features = self.conv(features).transpose(1, 2)
        features, _ = self.gru(features)
        mask = self.mask_head(features).transpose(1, 2)
        mask_real, mask_imag = mask.chunk(2, dim=1)
        # Unity mask at initialization; bounded corrections avoid unstable
        # amplification while allowing both restoration and phase adjustment.
        mask_real = 1.0 + 1.5 * torch.tanh(mask_real)
        mask_imag = 0.75 * torch.tanh(mask_imag)
        enhanced_real = real * mask_real - imag * mask_imag
        enhanced_imag = real * mask_imag + imag * mask_real
        return self.istft(enhanced_real, enhanced_imag, length=residual.shape[-1])
