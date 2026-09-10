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


def combined_loss(estimate: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.1) -> torch.Tensor:
    """SI-SNR + L1, matching the plan's suggested starting weights.
    (Perceptual loss is left as a TODO — wire in a pretrained-model-based
    perceptual loss, e.g. from torchaudio, once SI-SNR + L1 is converging.)"""
    return si_snr(estimate, target) + l1_weight * l1_loss(estimate, target)
