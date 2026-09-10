"""
Lightweight Conv-TasNet-style model — time-domain: a learned encoder,
a stack of dilated temporal-conv (TCN) blocks that predict a mask, and a
learned decoder. Captures phase implicitly (unlike STFT-magnitude models
like crn.py) and tends to do better on impulsive transients, at higher
compute cost — matches the trade-off table in the architecture doc.

This is a *reduced* version (fewer blocks/channels than the paper) sized
for embedded real-time use; scale tcn_blocks/tcn_repeats/tcn_channels up
if you have compute headroom and latency budget to spare.
"""
import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation  # kernel_size=3 -> padding=dilation keeps length constant
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding,
                      dilation=dilation, groups=channels),  # depthwise
            nn.PReLU(),
            nn.GroupNorm(1, channels),
            nn.Conv1d(channels, channels, kernel_size=1),   # pointwise
            nn.PReLU(),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvTasNetLite(nn.Module):
    def __init__(self, encoder_dim: int = 128, kernel_size: int = 16, stride: int = 8,
                 tcn_channels: int = 128, tcn_blocks: int = 6, tcn_repeats: int = 2):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size=kernel_size, stride=stride, bias=False)
        self.bottleneck = nn.Conv1d(encoder_dim, tcn_channels, kernel_size=1)

        blocks = []
        for _ in range(tcn_repeats):
            for b in range(tcn_blocks):
                blocks.append(TCNBlock(tcn_channels, dilation=2 ** b))
        self.tcn = nn.Sequential(*blocks)

        self.mask_head = nn.Sequential(
            nn.Conv1d(tcn_channels, encoder_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        x = waveform.unsqueeze(1)                 # (B, 1, T)
        encoded = self.encoder(x)                  # (B, encoder_dim, frames)

        feat = self.bottleneck(encoded)
        feat = self.tcn(feat)
        mask = self.mask_head(feat)

        masked = encoded * mask
        out = self.decoder(masked).squeeze(1)       # (B, T')

        # ConvTranspose1d output length can differ slightly from input length
        # depending on kernel/stride — pad or crop to match for the loss fn.
        if out.shape[-1] < waveform.shape[-1]:
            out = torch.nn.functional.pad(out, (0, waveform.shape[-1] - out.shape[-1]))
        else:
            out = out[..., :waveform.shape[-1]]
        return out
