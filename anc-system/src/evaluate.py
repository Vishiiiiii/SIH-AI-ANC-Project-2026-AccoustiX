"""
Evaluate a trained checkpoint on the test manifest, broken down by noise
category (stationary / non_stationary / impulsive) — this per-category
table is exactly what the plan calls for in section 7.

Usage:
    python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pt
"""
import argparse
import csv
from collections import defaultdict

import torch
import yaml
import soundfile as sf

from src.models.rnnoise import RNNoiseStyle
from src.models.conv_tasnet import ConvTasNetLite
from src.models.dtln import DTLN
from src.train_utils import si_snr

MODEL_REGISTRY = {"rnnoise": RNNoiseStyle, "conv_tasnet": ConvTasNetLite, "dtln": DTLN}

try:
    from pesq import pesq as pesq_fn
except ImportError:
    pesq_fn = None
try:
    from pystoi import stoi as stoi_fn
except ImportError:
    stoi_fn = None


def load_model(cfg, checkpoint_path, device):
    name = cfg["model"]
    params = cfg.get("model_params", {}).get(name, {})
    model = MODEL_REGISTRY[name](**params)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def evaluate(cfg, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, checkpoint_path, device)

    with open(cfg["data"]["test_manifest"]) as fh:
        rows = list(csv.DictReader(fh))

    per_category = defaultdict(list)
    with torch.no_grad():
        for row in rows:
            noisy, sr = sf.read(row["noisy_path"], dtype="float32")
            clean, _ = sf.read(row["clean_path"], dtype="float32")

            noisy_t = torch.from_numpy(noisy).unsqueeze(0).to(device)
            clean_t = torch.from_numpy(clean).unsqueeze(0).to(device)
            estimate_t = model(noisy_t)

            sisnr = -si_snr(estimate_t, clean_t).item()  # flip sign back to "higher is better"
            metrics = {"si_snr": sisnr}

            estimate = estimate_t.squeeze(0).cpu().numpy()
            if pesq_fn is not None and sr in (8000, 16000):
                try:
                    metrics["pesq"] = pesq_fn(sr, clean, estimate, "wb" if sr == 16000 else "nb")
                except Exception:
                    pass
            if stoi_fn is not None:
                try:
                    metrics["stoi"] = stoi_fn(clean, estimate, sr, extended=False)
                except Exception:
                    pass

            per_category[row["category"]].append(metrics)

    print(f"{'category':<15} {'n':>5} {'SI-SNR':>10} {'PESQ':>10} {'STOI':>10}")
    for category, results in sorted(per_category.items()):
        n = len(results)
        avg = lambda key: sum(r[key] for r in results if key in r) / max(1, sum(1 for r in results if key in r))
        print(f"{category:<15} {n:>5} {avg('si_snr'):>10.2f} "
              f"{avg('pesq') if any('pesq' in r for r in results) else float('nan'):>10.2f} "
              f"{avg('stoi') if any('stoi' in r for r in results) else float('nan'):>10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default=None,
                         help="Override the config's model: field to match the checkpoint being evaluated")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.model:
        cfg["model"] = args.model
    evaluate(cfg, args.checkpoint)
