"""
Precompute Wiener-filtered versions of an existing noisy/clean manifest,
so train.py can load them directly instead of running Wiener inline every
epoch (slow) or training on a mismatched raw-noisy distribution (see
realtime_demo.py note: the deployed pipeline runs Wiener before the model,
so the model must be trained on Wiener-filtered input to match).

Usage:
    python -m src.data.apply_wiener --manifest data/manifests/train.csv --out_dir data/processed_wiener/train
    python -m src.data.apply_wiener --manifest data/manifests/val.csv --out_dir data/processed_wiener/val
    python -m src.data.apply_wiener --manifest data/manifests/test.csv --out_dir data/processed_wiener/test
"""
import argparse
import csv
import pathlib

import soundfile as sf

from src.filters.wiener import WienerFilter


def apply_wiener_to_manifest(manifest_path: str, out_dir: str):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as fh:
        rows = list(csv.DictReader(fh))

    wf = WienerFilter()
    new_rows = []
    for row in rows:
        wf.reset()  # each clip is independent — don't leak noise history across clips
        noisy, sr = sf.read(row["noisy_path"])
        filtered = wf.process(noisy)

        out_name = pathlib.Path(row["noisy_path"]).name
        out_path = out_dir / out_name
        sf.write(out_path, filtered, sr)

        new_row = dict(row)
        new_row["noisy_path"] = str(out_path)   # clean_path stays the same — target is unchanged
        new_rows.append(new_row)

    out_manifest = out_dir.parent.parent / "manifests" / f"{out_dir.name}_wiener.csv"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(new_rows[0].keys()))
        writer.writeheader()
        writer.writerows(new_rows)
    print(f"Wrote {len(new_rows)} Wiener-filtered pairs -> {out_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    apply_wiener_to_manifest(args.manifest, args.out_dir)