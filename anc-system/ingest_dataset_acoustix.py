"""
One-off ingestion of the extra noise material found in the sibling
`dataset acoustix/` folder (not part of the original data/raw corpus) into
data/raw/noise, resampled/downmixed to the project's 16kHz mono standard and
reorganized into category-named subfolders so noise_manifest.py's
DIRECTORY_CATEGORY_MAP can tag them automatically.

Sources folded in:
  - SESA (Sound Events for Surveillance Applications): casual/gunshot/
    explosion/siren, already 16kHz mono WAV, class encoded in filename
    (e.g. "gunshot_014.wav").
  - Impact-set: bounce/explosion/footsteps/gunshots/knocks/punch/smash,
    44.1kHz mono WAV, class encoded as the parent folder name. "bounce" is
    skipped — it's not a noise type relevant to any of the 3 DRDO
    categories (a ball bouncing isn't stationary/non_stationary/impulsive
    defence noise), everything else is kept.
  - 7 loose gunshot mp3s (gs1..gs7.mp3) at the dataset acoustix root.

Safe to re-run: skips a destination file if it already exists.

Usage (from anc-system/):
    python ingest_dataset_acoustix.py
"""
import pathlib
import re

import numpy as np
import soundfile as sf
import librosa

SOURCE_ROOT = pathlib.Path("../dataset acoustix")
DEST_ROOT = pathlib.Path("data/raw/noise")
TARGET_SR = 16000

IMPACT_SET_CLASSES = ["explosion", "footsteps", "gunshots", "knocks", "punch", "smash"]
SESA_FILENAME_RE = re.compile(r"^([a-zA-Z]+)_\d+\.wav$")


def load_mono_16k(path: pathlib.Path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype("float32")
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    return data


def write_if_missing(dest: pathlib.Path, data: np.ndarray):
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, data, TARGET_SR)
    return True


def ingest_sesa():
    written = skipped = failed = 0
    for split in ("train", "test"):
        split_dir = SOURCE_ROOT / "SESA" / split
        for f in sorted(split_dir.glob("*.wav")):
            m = SESA_FILENAME_RE.match(f.name)
            if not m:
                print(f"  SESA: couldn't parse class from {f.name}, skipping")
                continue
            cls = m.group(1).lower()
            dest = DEST_ROOT / "sesa" / cls / f.name
            try:
                data = load_mono_16k(f)
            except Exception as exc:
                failed += 1
                print(f"  SESA: failed to read {f}: {exc}")
                continue
            if write_if_missing(dest, data):
                written += 1
            else:
                skipped += 1
    print(f"SESA: {written} written, {skipped} already present, {failed} failed")


def ingest_impact_set():
    written = skipped = failed = 0
    for cls in IMPACT_SET_CLASSES:
        for split in ("train", "test"):
            split_dir = SOURCE_ROOT / "Impact-set" / split / cls
            if not split_dir.exists():
                continue
            for f in sorted(split_dir.glob("*.wav")):
                dest = DEST_ROOT / "impact_set" / cls / f"{split}_{f.name}"
                try:
                    data = load_mono_16k(f)
                except Exception as exc:
                    failed += 1
                    print(f"  Impact-set: failed to read {f}: {exc}")
                    continue
                if write_if_missing(dest, data):
                    written += 1
                else:
                    skipped += 1
    print(f"Impact-set: {written} written, {skipped} already present, {failed} failed")


def ingest_loose_gunshots():
    written = skipped = failed = 0
    for f in sorted(SOURCE_ROOT.glob("gs*.mp3")):
        dest = DEST_ROOT / "impact_set" / "gunshots" / f"extra_{f.stem}.wav"
        try:
            data = load_mono_16k(f)
        except Exception as exc:
            failed += 1
            print(f"  loose gunshots: failed to read {f}: {exc}")
            continue
        if write_if_missing(dest, data):
            written += 1
        else:
            skipped += 1
    print(f"Loose gunshot mp3s: {written} written, {skipped} already present, {failed} failed")


if __name__ == "__main__":
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"{SOURCE_ROOT.resolve()} not found — expected it as a sibling of anc-system/")
    ingest_sesa()
    ingest_impact_set()
    ingest_loose_gunshots()
    print("\nDone. Now regenerate the noise manifest:")
    print("  python -m src.data.noise_manifest --noise_dir data/raw/noise --out_csv data/manifests/noise_manifest.csv")
