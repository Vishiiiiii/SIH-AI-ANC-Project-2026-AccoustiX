"""
PyTorch Dataset over a (noisy_path, clean_path) manifest CSV produced by
build_dataset.py.

Bug fix vs. the pasted snippet: the dunder methods were typed as single
underscores (`_init_`, `_len_`, `_getitem_`) — WhatsApp/markdown ate the
double underscores when it was shared. As written, PyTorch's DataLoader
would silently fall back to default object behavior and crash or hang
rather than actually loading data. Fixed below.
"""
import csv
import torch
import soundfile as sf
from torch.utils.data import Dataset


def load_manifest(csv_path):
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    return [(r["noisy_path"], r["clean_path"]) for r in rows]


def load_dual_mic_manifest(csv_path):
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"noisy_path", "raw_noisy_path", "nlms_path", "clean_path"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Dual-mic manifest is missing {sorted(missing)}: {csv_path}")
    return [(r["noisy_path"], r["raw_noisy_path"], r["nlms_path"], r["clean_path"]) for r in rows]


class NoisyCleanDataset(Dataset):
    def __init__(self, manifest_csv, segment_len=16000 * 4, random_crop=True):
        self.manifest = load_manifest(manifest_csv)
        self.segment_len = segment_len
        self.random_crop = random_crop

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.manifest[idx]
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")

        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)

        if len(noisy) < self.segment_len:
            pad = self.segment_len - len(noisy)
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))
        else:
            # random crop keeps every epoch seeing a different slice of long clips
            choices = len(noisy) - self.segment_len + 1
            start = (torch.randint(0, choices, (1,)).item() if self.random_crop
                     else (idx * 7919) % choices)
            noisy = noisy[start:start + self.segment_len]
            clean = clean[start:start + self.segment_len]

        return noisy, clean


class DualMicNoisyCleanDataset(Dataset):
    """Residual + aligned-noise feature pairs for the learned dual-mic stage."""
    def __init__(self, manifest_csv, segment_len=16000 * 4, include_primary=False,
                 random_crop=True):
        self.manifest = load_dual_mic_manifest(manifest_csv)
        self.segment_len = segment_len
        self.include_primary = include_primary
        self.random_crop = random_crop

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        residual_path, raw_path, nlms_path, clean_path = self.manifest[idx]
        residual, _ = sf.read(residual_path, dtype="float32")
        raw, _ = sf.read(raw_path, dtype="float32")
        nlms, _ = sf.read(nlms_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        residual = torch.from_numpy(residual)
        # NLMS residual = raw primary - aligned predicted noise.
        noise_estimate = torch.from_numpy(raw - nlms)
        raw = torch.from_numpy(raw)
        clean = torch.from_numpy(clean)
        if len(residual) < self.segment_len:
            pad = self.segment_len - len(residual)
            residual = torch.nn.functional.pad(residual, (0, pad))
            noise_estimate = torch.nn.functional.pad(noise_estimate, (0, pad))
            raw = torch.nn.functional.pad(raw, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))
        else:
            choices = len(residual) - self.segment_len + 1
            start = (torch.randint(0, choices, (1,)).item() if self.random_crop
                     else (idx * 7919) % choices)
            residual = residual[start:start + self.segment_len]
            noise_estimate = noise_estimate[start:start + self.segment_len]
            raw = raw[start:start + self.segment_len]
            clean = clean[start:start + self.segment_len]
        features = (residual, noise_estimate, raw) if self.include_primary else (residual, noise_estimate)
        return torch.stack(features), clean
