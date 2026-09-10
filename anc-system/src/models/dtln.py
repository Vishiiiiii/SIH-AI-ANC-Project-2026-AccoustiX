"""
DTLN — Dual-signal Transformation LSTM Network (simplified), after
Westhausen & Meyer (2020). Designed specifically for real-time, low-latency
embedded enhancement, which is why the plan lists it as a natural middle
ground between rnnoise.py (cheap, frequency-domain, limited on transients)
and conv_tasnet.py (better quality, heavier compute).

Two stages, each causal (no future look-ahead):
  Stage 1 (frequency domain): STFT magnitude -> 2-layer LSTM -> mask.
          Cheap, handles most stationary/non-stationary noise.
  Stage 2 (time domain): a small learned encoder/decoder + LSTM refines
          the stage-1 output directly on the waveform, correcting phase
          and residual/impulsive artifacts stage 1 can't reach.
"""
import torch
import torch.nn as nn

from src.models.stft_utils import ConvSTFT, ConvISTFT


class DTLN(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 lstm1_hidden: int = 128, encoder_dim: int = 256,
                 encoder_kernel: int = 32, lstm2_hidden: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        freq_bins = n_fft // 2 + 1

        # --- Stage 1: frequency-domain masking (conv-based STFT — see
        # stft_utils.py; ONNX's STFT op can't represent complex tensors) ---
        self.stft = ConvSTFT(n_fft, hop_length)
        self.istft = ConvISTFT(n_fft, hop_length)
        self.lstm1 = nn.LSTM(input_size=freq_bins, hidden_size=lstm1_hidden,
                              num_layers=2, batch_first=True)
        self.mask1_head = nn.Sequential(nn.Linear(lstm1_hidden, freq_bins), nn.Sigmoid())

        # --- Stage 2: time-domain refinement ---
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size=encoder_kernel,
                                  stride=encoder_kernel // 2, bias=False)
        self.lstm2 = nn.LSTM(input_size=encoder_dim, hidden_size=lstm2_hidden,
                              num_layers=2, batch_first=True)
        self.mask2_head = nn.Sequential(nn.Linear(lstm2_hidden, encoder_dim), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=encoder_kernel,
                                           stride=encoder_kernel // 2, bias=False)

    def _stage1(self, waveform: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(waveform)             # (B, F, frames) each
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        x = mag.transpose(1, 2)                    # (B, frames, F)
        x, _ = self.lstm1(x)
        mask = self.mask1_head(x).transpose(1, 2)   # (B, F, frames)

        # masking (real, imag) directly == masking magnitude while
        # preserving phase (see stft_utils.py docstring)
        stage1_out = self.istft(real * mask, imag * mask, length=waveform.shape[-1])
        return stage1_out

    def _stage2(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)                  # (B, 1, T)
        encoded = self.encoder(x)                    # (B, encoder_dim, frames)

        feat = encoded.transpose(1, 2)               # (B, frames, encoder_dim)
        feat, _ = self.lstm2(feat)
        mask = self.mask2_head(feat).transpose(1, 2)  # (B, encoder_dim, frames)

        masked = encoded * mask
        out = self.decoder(masked).squeeze(1)
        if out.shape[-1] < waveform.shape[-1]:
            out = torch.nn.functional.pad(out, (0, waveform.shape[-1] - out.shape[-1]))
        else:
            out = out[..., :waveform.shape[-1]]
        return out

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        stage1_out = self._stage1(waveform)
        stage2_out = self._stage2(stage1_out)
        return stage2_out
