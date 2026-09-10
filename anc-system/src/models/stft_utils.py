"""
ONNX-exportable STFT/ISTFT as fixed (non-trainable) Conv1d / ConvTranspose1d
layers, instead of torch.stft/istft.

Why: ONNX's STFT operator doesn't support complex-valued tensors, and
torch.stft(..., return_complex=True) can't be traced through the legacy
TorchScript ONNX exporter as a result ("STFT does not currently support
complex types"). Representing the DFT as a convolution with fixed
cos/sin-windowed kernels sidesteps that entirely, and is standard practice
for real-time/embedded audio models (it also maps naturally onto streaming
inference later, since each output frame only depends on a fixed window of
past+current samples).

Math: since our downstream models only ever multiply the STFT magnitude by
a real, non-negative mask (preserving phase), masking the complex STFT is
equivalent to masking (real, imag) directly: no explicit phase/atan2 needed
anywhere, which keeps the whole thing free of ONNX-unfriendly trig ops.
"""
import math
import torch
import torch.nn as nn


class ConvSTFT(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        freq_bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, periodic=True)
        n = torch.arange(n_fft).float()
        k = torch.arange(freq_bins).float().unsqueeze(1)
        theta = 2 * math.pi * k * n / n_fft                      # (freq_bins, n_fft)

        weight_real = (window * torch.cos(theta)).unsqueeze(1)   # (freq_bins, 1, n_fft)
        weight_imag = (-window * torch.sin(theta)).unsqueeze(1)

        self.register_buffer("weight_real", weight_real)
        self.register_buffer("weight_imag", weight_imag)

    def forward(self, waveform: torch.Tensor):
        # waveform: (B, T) -> real, imag: (B, freq_bins, frames)
        x = waveform.unsqueeze(1)
        x = torch.nn.functional.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        real = torch.nn.functional.conv1d(x, self.weight_real, stride=self.hop_length)
        imag = torch.nn.functional.conv1d(x, self.weight_imag, stride=self.hop_length)
        return real, imag


class ConvISTFT(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        freq_bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, periodic=True)
        n = torch.arange(n_fft).float()
        k = torch.arange(freq_bins).float().unsqueeze(1)
        theta = 2 * math.pi * k * n / n_fft

        scale = torch.full((freq_bins, 1), 2.0)
        scale[0, 0] = 1.0
        if n_fft % 2 == 0:
            scale[-1, 0] = 1.0
        scale = scale / n_fft

        inv_weight_real = (scale * torch.cos(theta)).unsqueeze(1)    # (freq_bins, 1, n_fft)
        inv_weight_imag = (-scale * torch.sin(theta)).unsqueeze(1)

        self.register_buffer("inv_weight_real", inv_weight_real)
        self.register_buffer("inv_weight_imag", inv_weight_imag)
        # Norm kernel: overlap-adding the analysis window itself gives the
        # COLA normalization envelope (see module docstring's derivation).
        self.register_buffer("norm_kernel", window.view(1, 1, n_fft))

    def forward(self, real: torch.Tensor, imag: torch.Tensor, length: int):
        # real, imag: (B, freq_bins, frames) -> waveform: (B, length)
        out = (torch.nn.functional.conv_transpose1d(real, self.inv_weight_real, stride=self.hop_length)
               + torch.nn.functional.conv_transpose1d(imag, self.inv_weight_imag, stride=self.hop_length))

        frames = real.shape[-1]
        ones = torch.ones(1, 1, frames, device=real.device, dtype=real.dtype)
        norm = torch.nn.functional.conv_transpose1d(ones, self.norm_kernel, stride=self.hop_length)
        out = out / (norm + 1e-8)

        out = out.squeeze(1)
        pad = self.n_fft // 2
        out = out[:, pad:out.shape[-1] - pad]  # remove the reflect-pad added at analysis time

        if out.shape[-1] < length:
            out = torch.nn.functional.pad(out, (0, length - out.shape[-1]))
        else:
            out = out[:, :length]
        return out
