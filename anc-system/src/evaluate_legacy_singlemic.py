"""Approximate evaluation for the archived single-mic Wiener + RNNoise model.

The original ``data/processed_wiener`` files used to train this model are no
longer present. This script reconstructs the model input from the original
raw held-out test pairs with the project Wiener implementation. Therefore its
numbers are useful for comparison, but are not a byte-for-byte historical
reproduction of the missing precomputed inputs.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

from src.evaluate import averages, calculate_metrics, load_model, print_categories, print_overall
from src.filters.wiener import WienerFilter
from src.train_utils import si_snr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/test.csv")
    parser.add_argument("--noise-estimate-ms", type=float, default=300,
                        help="Legacy Wiener default used by the original helper")
    parser.add_argument("--si-snr-only", action="store_true",
                        help="Skip slow PESQ/STOI and calculate full-test SI-SNR only")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    wiener = WienerFilter(noise_estimate_ms=args.noise_estimate_ms)
    with open(args.manifest, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    stages = ("Raw primary", "Wiener only", "Wiener + archived RNNoise")
    all_results = {stage: [] for stage in stages}
    by_category = {stage: defaultdict(list) for stage in stages}
    print(f"Device: {device}; examples: {len(rows)}; Wiener estimate: {args.noise_estimate_ms:g} ms")
    print("NOTE: reconstructed Wiener input; the original precomputed Wiener WAVs are unavailable.")

    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            raw, raw_sr = sf.read(row["noisy_path"], dtype="float32")
            clean, clean_sr = sf.read(row["clean_path"], dtype="float32")
            if raw_sr != clean_sr:
                raise RuntimeError(f"Sample-rate mismatch in row {index}.")
            filtered = wiener.process(raw, raw_sr)
            enhanced = model(torch.from_numpy(filtered).unsqueeze(0).to(device))
            enhanced = enhanced.squeeze(0).cpu().numpy().astype(np.float32)
            for stage, signal in zip(stages, (raw, filtered, enhanced), strict=True):
                if args.si_snr_only:
                    clean_t = torch.from_numpy(clean).unsqueeze(0)
                    signal_t = torch.from_numpy(signal).unsqueeze(0)
                    metrics = {"si_snr": float(-si_snr(signal_t, clean_t).item())}
                else:
                    metrics = calculate_metrics(clean, signal, clean_sr)
                all_results[stage].append(metrics)
                by_category[stage][row["category"]].append(metrics)
            if index % 100 == 0 or index == len(rows):
                print(f"Evaluated {index}/{len(rows)}")

    print("\n" + "=" * 70 + "\nArchived single-mic test results (approximate)\n" + "=" * 70)
    for stage in stages:
        print_overall(stage, all_results[stage])
    raw = averages(all_results["Raw primary"])
    hybrid = averages(all_results["Wiener + archived RNNoise"])
    print(f"\nArchived hybrid vs raw: SI-SNR {hybrid['si_snr'] - raw['si_snr']:+.2f} dB | "
          f"PESQ {hybrid['pesq'] - raw['pesq']:+.3f} | STOI {hybrid['stoi'] - raw['stoi']:+.3f}")
    for stage in stages:
        print_categories(f"{stage.upper()} BY CATEGORY", by_category[stage])


if __name__ == "__main__":
    main()
