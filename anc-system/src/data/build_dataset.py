"""
Generate (noisy, clean) training pairs from a speech directory + a tagged
noise manifest, at randomized SNRs, split into train/val/test.

Usage:
    python -m src.data.build_dataset \
        --speech_dir data/raw/speech --noise_manifest data/manifests/noise_manifest.csv \
        --out_dir data/processed --n_train 4000 --n_val 500 --n_test 500
"""
import argparse
import csv
import pathlib
import numpy as np
from collections import defaultdict

import soundfile as sf

from src.data.mix import mix_at_snr, randomize_impulsive_onset, synthesize_reference

DEFAULT_SNRS = [-5, 0, 5, 10, 15]


def load_noise_manifest(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def find_speech_files(speech_dir):
    return [str(p) for p in pathlib.Path(speech_dir).rglob("*.flac")] + \
           [str(p) for p in pathlib.Path(speech_dir).rglob("*.wav")]


def split_noise_by_category(noise_rows, seed=0, train_frac=0.8, val_frac=0.1):
    """Create disjoint, category-stratified source pools for each split.

    Sampling may reuse a source clip within a split, but the same underlying
    noise recording can never occur in both training and evaluation.
    """
    rng = np.random.default_rng(seed)
    by_category = defaultdict(list)
    for row in noise_rows:
        by_category[row["category"]].append(row)

    pools = {"train": [], "val": [], "test": []}
    for category, rows in by_category.items():
        indices = rng.permutation(len(rows))
        shuffled = [rows[int(index)] for index in indices]
        n_train = max(1, int(train_frac * len(shuffled)))
        n_val = max(1, int(val_frac * len(shuffled))) if len(shuffled) - n_train > 1 else 0
        pools["train"].extend(shuffled[:n_train])
        pools["val"].extend(shuffled[n_train:n_train + n_val])
        pools["test"].extend(shuffled[n_train + n_val:])
    if not all(pools.values()):
        raise ValueError("A split has no noise clips; add more labeled source audio.")
    return pools["train"], pools["val"], pools["test"]


def build_split(split_name, n_examples, speech_files, noise_rows, out_dir, snrs, seed,
                manifest_dir, with_reference=False, balanced_categories=True,
                clip_seconds=None):
    rng = np.random.default_rng(seed)
    split_dir = pathlib.Path(out_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rows_by_category = {}
    for row in noise_rows:
        rows_by_category.setdefault(row["category"], []).append(row)
    categories = sorted(rows_by_category)
    for i in range(n_examples):
        speech_path = str(rng.choice(speech_files))
        if balanced_categories:
            category = str(rng.choice(categories))
            candidates = rows_by_category[category]
            noise_row = candidates[int(rng.integers(0, len(candidates)))]
        else:
            noise_row = noise_rows[int(rng.integers(0, len(noise_rows)))]
        snr = int(rng.choice(snrs))

        clean, sr = sf.read(speech_path)
        noise, noise_sr = sf.read(noise_row["path"])
        if noise_sr != sr:
            raise ValueError(
                f"Sample-rate mismatch: {speech_path} @ {sr}Hz vs {noise_row['path']} @ {noise_sr}Hz. "
                "Resample your corpora to a common rate (16kHz recommended) before mixing."
            )

        if clip_seconds is not None:
            clip_samples = int(round(clip_seconds * sr))
            if len(clean) >= clip_samples:
                start = int(rng.integers(0, len(clean) - clip_samples + 1))
                clean = clean[start:start + clip_samples]
            else:
                # Preserve the speech once rather than repeating it. The short
                # zero tail is preferable to teaching the model repeated words.
                pad_width = [(0, clip_samples - len(clean))] + [(0, 0)] * (clean.ndim - 1)
                clean = np.pad(clean, pad_width)

        if noise_row["category"] == "impulsive":
            noise = randomize_impulsive_onset(noise, len(clean), rng=rng)
        
        noisy, clean, noise_component = mix_at_snr(
            clean, noise, snr, rng=rng, return_noise=True,
        )

        noisy_path = split_dir / f"noisy_{i:05d}.wav"
        clean_path = split_dir / f"clean_{i:05d}.wav"
        sf.write(noisy_path, noisy, sr)
        sf.write(clean_path, clean, sr)

        row = {
            "noisy_path": str(noisy_path), "clean_path": str(clean_path),
            "category": noise_row["category"], "snr_db": snr,
            "speech_source": speech_path, "noise_source": noise_row["path"],
        }
        if with_reference:
            reference, reference_info = synthesize_reference(noise_component, rng, sample_rate=sr)
            reference_path = split_dir / f"reference_{i:05d}.wav"
            sf.write(reference_path, reference, sr)
            row["reference_path"] = str(reference_path)
            row.update(reference_info)
        manifest_rows.append(row)

    manifest_out = pathlib.Path(manifest_dir) / f"{split_name}.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"{split_name}: wrote {len(manifest_rows)} pairs -> {manifest_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech_dir", required=True)
    parser.add_argument("--noise_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_train", type=int, default=4000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--snrs", type=int, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest_dir", default=None,
                        help="Output directory for split CSVs (default: <out_dir>/../manifests)")
    parser.add_argument("--with_reference", action="store_true",
                        help="Write a synthetic correlated reference-mic WAV for every pair")
    parser.add_argument("--natural_category_distribution", action="store_true",
                        help="Sample by source-file count instead of the default balanced categories")
    parser.add_argument("--clip_seconds", type=float, default=4.0,
                        help="Fixed pair duration; use 0 to preserve full speech files")
    args = parser.parse_args()

    speech_files = find_speech_files(args.speech_dir)
    noise_rows = load_noise_manifest(args.noise_manifest)
    if not speech_files:
        raise SystemExit(f"No .wav/.flac files found under {args.speech_dir}")
    if not noise_rows:
        raise SystemExit(f"Noise manifest {args.noise_manifest} is empty")

    # Drop unlabeled clips — an "unknown" category is meaningless for
    # per-category evaluation downstream, so don't let them leak into training.
    n_before = len(noise_rows)
    noise_rows = [r for r in noise_rows if r["category"] != "unknown"]
    n_dropped = n_before - len(noise_rows)
    if n_dropped:
        print(f"Dropped {n_dropped}/{n_before} noise clips with category='unknown'.")
    if not noise_rows:
        raise SystemExit(
            "All noise clips are labeled 'unknown' after filtering — check "
            "KEYWORD_CATEGORY_MAP / ESC50_CATEGORY_MAP in noise_manifest.py."
        )

    # IMPORTANT: split speech files (and ideally noise clips too) BEFORE
    # sampling, so the same speaker/clip never appears in both train and test.
    rng = np.random.default_rng(args.seed)
    rng.shuffle(speech_files)
    n = len(speech_files)
    train_speech = speech_files[: int(0.8 * n)] or speech_files
    val_speech = speech_files[int(0.8 * n): int(0.9 * n)] or speech_files
    test_speech = speech_files[int(0.9 * n):] or speech_files

    manifest_dir = args.manifest_dir or str(pathlib.Path(args.out_dir).parent / "manifests")
    clip_seconds = args.clip_seconds if args.clip_seconds > 0 else None
    train_noise, val_noise, test_noise = split_noise_by_category(noise_rows, seed=args.seed)
    print(f"Noise source pools: train={len(train_noise)}, val={len(val_noise)}, test={len(test_noise)}")
    build_split("train", args.n_train, train_speech, train_noise, args.out_dir, args.snrs, args.seed,
                manifest_dir, args.with_reference, not args.natural_category_distribution, clip_seconds)
    build_split("val", args.n_val, val_speech, val_noise, args.out_dir, args.snrs, args.seed + 1,
                manifest_dir, args.with_reference, not args.natural_category_distribution, clip_seconds)
    build_split("test", args.n_test, test_speech, test_noise, args.out_dir, args.snrs, args.seed + 2,
                manifest_dir, args.with_reference, not args.natural_category_distribution, clip_seconds)
