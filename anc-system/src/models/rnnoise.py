"""
RNNoiseStyle — a small Convolutional-Recurrent mask estimator on the STFT
magnitude. This is the "RNNoise-style, frequency-domain" baseline: much
simpler than real RNNoise's hand-crafted Bark-scale features + GRU, but
trainable end-to-end in PyTorch, ONNX-exportable, and a fair stand-in to
get a real-time-safe demo working first. Swap in true RNNoise (via its C
implementation + Python bindings) later if you need the exact ~85k-param
embedded footprint.

STFT/ISTFT are implemented as fixed Conv1d/ConvTranspose1d layers (see
stft_utils.py) rather than torch.stft/istft, so this exports to ONNX
cleanly (ONNX's STFT op can't represent complex tensors).

NOTE ON CAUSALITY: this processes a whole clip/block at once (fine for
training and for the block-based real-time demo). For a fully causal
streaming version, the GRU hidden state would need to be carried across
calls explicitly — see realtime_demo.py's TODOs.
"""
import torch
import torch.nn as nn

from src.models.stft_utils import ConvSTFT, ConvISTFT


class RNNoiseStyle(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128, gru_hidden: int = 128):
        super().__init__()
        freq_bins = n_fft // 2 + 1
        self.stft = ConvSTFT(n_fft, hop_length)
        self.istft = ConvISTFT(n_fft, hop_length)

        self.conv = nn.Sequential(
            nn.Conv1d(freq_bins, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.PReLU(),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden, batch_first=True)
        self.mask_head = nn.Sequential(
            nn.Linear(gru_hidden, freq_bins),
            nn.Sigmoid(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(waveform)                # (B, F, frames) each
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        x = self.conv(mag)                       # (B, 128, frames)
        x = x.transpose(1, 2)                     # (B, frames, 128)
        x, _ = self.gru(x)                        # (B, frames, gru_hidden)
        mask = self.mask_head(x).transpose(1, 2)   # (B, F, frames)

        # Masking magnitude while preserving phase == masking (real, imag)
        # directly by the same real mask — no explicit phase needed.
        enhanced = self.istft(real * mask, imag * mask, length=waveform.shape[-1])
        return enhanced
