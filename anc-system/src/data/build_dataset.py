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
import random

import soundfile as sf

from src.data.mix import mix_at_snr, randomize_impulsive_onset

DEFAULT_SNRS = [-5, 0, 5, 10, 15]


def load_noise_manifest(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def find_speech_files(speech_dir):
    return [str(p) for p in pathlib.Path(speech_dir).rglob("*.flac")] + \
           [str(p) for p in pathlib.Path(speech_dir).rglob("*.wav")]


def build_split(split_name, n_examples, speech_files, noise_rows, out_dir, snrs, seed):
    rng = random.Random(seed)
    split_dir = pathlib.Path(out_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Sample category first, then a clip within it — NOT rng.choice(noise_rows)
    # directly. The raw noise pool is unevenly sized per category (e.g. far
    # more non_stationary clips than stationary), so choosing rows uniformly
    # would silently skew the mixed dataset toward whichever category has the
    # most source recordings instead of giving stationary/non_stationary/
    # impulsive equal representation — which matters since that 3-way split
    # is exactly what the DRDO problem statement asks for and what
    # evaluate.py reports on per-category.
    by_category = {}
    for row in noise_rows:
        by_category.setdefault(row["category"], []).append(row)
    categories = sorted(by_category)

    manifest_rows = []
    bad_files = set()
    for i in range(n_examples):
        # A handful of source files (e.g. some FreeNoise .mp3 clips libsndfile
        # can't decode) are unreadable — skip and retry with a different pick
        # rather than letting one bad file kill an otherwise-fine 4000-example
        # run. Capped attempts so a systematically broken corpus still fails
        # loudly instead of spinning forever.
        for attempt in range(20):
            speech_path = rng.choice(speech_files)
            noise_row = rng.choice(by_category[rng.choice(categories)])
            snr = rng.choice(snrs)
            try:
                clean, sr = sf.read(speech_path)
            except Exception as exc:
                bad_files.add((speech_path, str(exc)))
                continue
            try:
                noise, noise_sr = sf.read(noise_row["path"])
            except Exception as exc:
                bad_files.add((noise_row["path"], str(exc)))
                continue
            if noise_sr != sr:
                raise ValueError(
                    f"Sample-rate mismatch: {speech_path} @ {sr}Hz vs {noise_row['path']} @ {noise_sr}Hz. "
                    "Resample your corpora to a common rate (16kHz recommended) before mixing."
                )
            break
        else:
            raise RuntimeError(
                f"Couldn't find a readable speech/noise pair after 20 attempts at example {i} "
                f"of split '{split_name}' — check data/raw for widespread corrupt/unsupported files."
            )

        if noise_row["category"] == "impulsive":
            noise = randomize_impulsive_onset(noise, len(clean))

        noisy, clean = mix_at_snr(clean, noise, snr)

        noisy_path = split_dir / f"noisy_{i:05d}.wav"
        clean_path = split_dir / f"clean_{i:05d}.wav"
        sf.write(noisy_path, noisy, sr)
        sf.write(clean_path, clean, sr)

        manifest_rows.append({
            "noisy_path": str(noisy_path), "clean_path": str(clean_path),
            "category": noise_row["category"], "snr_db": snr,
            "speech_source": speech_path, "noise_source": noise_row["path"],
        })

    manifest_out = pathlib.Path(out_dir).parent / "manifests" / f"{split_name}.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"{split_name}: wrote {len(manifest_rows)} pairs -> {manifest_out}")
    if bad_files:
        print(f"{split_name}: skipped {len(bad_files)} unreadable source file(s):")
        for path, err in sorted(bad_files):
            print(f"    {path}  ({err})")


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
    rng = random.Random(args.seed)
    rng.shuffle(speech_files)
    n = len(speech_files)
    train_speech = speech_files[: int(0.8 * n)] or speech_files
    val_speech = speech_files[int(0.8 * n): int(0.9 * n)] or speech_files
    test_speech = speech_files[int(0.9 * n):] or speech_files

    build_split("train", args.n_train, train_speech, noise_rows, args.out_dir, args.snrs, args.seed)
    build_split("val", args.n_val, val_speech, noise_rows, args.out_dir, args.snrs, args.seed + 1)
    build_split("test", args.n_test, test_speech, noise_rows, args.out_dir, args.snrs, args.seed + 2)
