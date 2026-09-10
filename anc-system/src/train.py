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

from src.data.dataset import NoisyCleanDataset
from src.models.rnnoise import RNNoiseStyle
from src.models.conv_tasnet import ConvTasNetLite
from src.models.dtln import DTLN
from src.train_utils import combined_loss

MODEL_REGISTRY = {"rnnoise": RNNoiseStyle, "conv_tasnet": ConvTasNetLite, "dtln": DTLN}


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

    train_ds = NoisyCleanDataset(cfg["data"]["train_manifest"], segment_len=segment_len)
    val_ds = NoisyCleanDataset(cfg["data"]["val_manifest"], segment_len=segment_len)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], num_workers=2)

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    ckpt_dir = pathlib.Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for noisy, clean in tqdm(train_dl, desc=f"epoch {epoch} [train]"):
            noisy, clean = noisy.to(device), clean.to(device)
            estimate = model(noisy)
            loss = combined_loss(estimate, clean, cfg["train"]["l1_weight"])

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                val_loss += combined_loss(model(noisy), clean, cfg["train"]["l1_weight"]).item()
        val_loss /= len(val_dl)

        print(f"epoch {epoch}: train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")

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
