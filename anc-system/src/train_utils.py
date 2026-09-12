"""
Loss functions and small training helpers.

Bug fix vs. the pasted snippet: `e_noise * 2` and `s_target * 2` were meant
to be `e_noise ** 2` and `s_target ** 2` (sum of SQUARES, i.e. signal power)
— `* 2` just doubles the values instead of squaring them, which makes the
SI-SNR ratio wrong (and not even always positive), so the loss would have
optimized against a broken objective. Fixed below, with a couple of other
small robustness additions (proper zero-mean pairing, epsilon placement).
"""
import torch


def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR loss (negative SI-SNR, so lower = better separation)."""
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    s_target = (torch.sum(estimate * target, dim=-1, keepdim=True) * target
                / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps))
    e_noise = estimate - s_target

    ratio = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return -10 * torch.log10(ratio + eps).mean()


def l1_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(estimate, target)


def multi_resolution_spectral_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Magnitude and log-magnitude error at speech-relevant STFT resolutions.

    PESQ is not differentiable, so it cannot be used as the training loss.
    This differentiable proxy penalizes the spectral smearing that strongly
    affects PESQ while SI-SNR alone can overlook it.
    """
    loss = estimate.new_zeros(())
    for n_fft in (256, 512):
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=estimate.device, dtype=estimate.dtype)
        est_spec = torch.stft(estimate, n_fft=n_fft, hop_length=hop, window=window,
                              return_complex=True, center=True).abs()
        target_spec = torch.stft(target, n_fft=n_fft, hop_length=hop, window=window,
                                 return_complex=True, center=True).abs()
        loss = loss + torch.nn.functional.l1_loss(est_spec, target_spec)
        loss = loss + torch.nn.functional.l1_loss(torch.log1p(est_spec), torch.log1p(target_spec))
    return loss / 2


def combined_loss(estimate: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.1,
                  spectral_weight: float = 0.0) -> torch.Tensor:
    """SI-SNR with optional time- and frequency-domain fidelity terms."""
    return (si_snr(estimate, target) + l1_weight * l1_loss(estimate, target)
            + spectral_weight * multi_resolution_spectral_loss(estimate, target))
