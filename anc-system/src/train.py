"""
Training entrypoint, wired to configs/default.yaml.

Usage:
    python -m src.train --config configs/default.yaml
"""
import argparse
import pathlib

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NoisyCleanDataset, DualMicNoisyCleanDataset
from src.models.rnnoise import RNNoiseStyle
from src.models.conv_tasnet import ConvTasNetLite
from src.models.dtln import DTLN
from src.models.dual_mic_rnnoise import DualMicRNNoiseStyle
from src.models.dual_mic_complex_rnnoise import DualMicComplexRNNoiseStyle
from src.train_utils import combined_loss

MODEL_REGISTRY = {"rnnoise": RNNoiseStyle, "conv_tasnet": ConvTasNetLite, "dtln": DTLN,
                  "dual_mic_rnnoise": DualMicRNNoiseStyle}
MODEL_REGISTRY["dual_mic_complex_rnnoise"] = DualMicComplexRNNoiseStyle


def build_model(cfg):
    name = cfg["model"]
    params = cfg.get("model_params", {}).get(name, {})
    return MODEL_REGISTRY[name](**params)


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def train(cfg):
    device = get_device(cfg["train"]["device"])
    segment_len = int(cfg["sample_rate"] * cfg["segment_seconds"])

    input_mode = cfg["data"].get("input")
    if input_mode in {"dual_mic", "dual_mic_primary_context"}:
        include_primary = input_mode == "dual_mic_primary_context"
        train_ds = DualMicNoisyCleanDataset(cfg["data"]["train_manifest"], segment_len=segment_len,
                                             include_primary=include_primary)
        val_ds = DualMicNoisyCleanDataset(cfg["data"]["val_manifest"], segment_len=segment_len,
                                           include_primary=include_primary, random_crop=False)
    else:
        train_ds = NoisyCleanDataset(cfg["data"]["train_manifest"], segment_len=segment_len)
        val_ds = NoisyCleanDataset(cfg["data"]["val_manifest"], segment_len=segment_len,
                                   random_crop=False)
    # Multiprocess loaders are faster on an unrestricted machine, but Windows
    # sandboxed environments can reject multiprocessing.Queue creation. Keep
    # it configurable so the same code works in both places.
    num_workers = cfg["train"].get("num_workers", 0)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=num_workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                        num_workers=num_workers, pin_memory=device.type == "cuda")

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg["train"].get("lr_decay", 0.5),
        patience=cfg["train"].get("lr_patience", 5),
    )

    ckpt_dir = pathlib.Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for noisy, clean in tqdm(train_dl, desc=f"epoch {epoch} [train]"):
            noisy, clean = noisy.to(device), clean.to(device)
            estimate = model(noisy)
            loss = combined_loss(estimate, clean, cfg["train"]["l1_weight"],
                                 cfg["train"].get("spectral_weight", 0.0))

            opt.zero_grad()
            loss.backward()
            grad_clip = cfg["train"].get("gradient_clip_norm")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                val_loss += combined_loss(model(noisy), clean, cfg["train"]["l1_weight"],
                                           cfg["train"].get("spectral_weight", 0.0)).item()
        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        print(f"epoch {epoch}: train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg},
                       ckpt_dir / "best_model.pt")
            print(f"  -> new best, saved to {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    train(cfg)
