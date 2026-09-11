"""

Tag raw noise clips by category (stationary / non_stationary / impulsive)

so downstream mixing + evaluation can report per-category metrics.



Supports:

  1. FreeNoise directory: Multi-format audio (.wav, .flac, .mp4, etc.) categorized via keywords.

  2. ESC-50 dataset: Uses esc50.csv to resolve filenames and maps ESC-50 classes to broader noise categories.



Usage:

    python -m src.data.noise_manifest --noise_dir data/raw/noise --out_csv data/manifests/noise_manifest.csv

"""

import argparse

import csv

import pathlib

from typing import Dict



# Keyword matching for FreeNoise / unstructured files

KEYWORD_CATEGORY_MAP = {

    "engine": "stationary", "hvac": "stationary", "hum": "stationary",

    "fan": "stationary", "generator": "stationary",

    "wind": "non_stationary", "chatter": "non_stationary", "rotor": "non_stationary",

    "crowd": "non_stationary", "traffic": "non_stationary", "rain": "non_stationary",

    "gunfires": "impulsive", "explosions": "impulsive", "artillery": "impulsive",

    "gunshots": "impulsive", "blast": "impulsive",

}



# Mapping ESC-50 ground truth categories to high-level noise profiles

ESC50_CATEGORY_MAP = {

    # Impulsive

    "fireworks": "impulsive",

    "door_wood_knock": "impulsive",

    "glass_breaking": "impulsive",

    "car_horn": "impulsive",

    # Stationary

    "engine": "stationary",

    "airplane": "stationary",

    "train": "stationary",

    "siren": "stationary",

    # Non-Stationary

    "rain": "non_stationary",

    "crackling_fire": "non_stationary",

    "wind": "non_stationary",

    "pouring_water": "non_stationary",

    "footsteps": "non_stationary",

    "breathing": "non_stationary",

    "coughing": "non_stationary",

    "helicopter": "non_stationary",

    "thunderstorm": "non_stationary",

    "door_wood_creaks": "non_stationary",

}



AUDIO_EXTENSIONS = {".wav", ".flac", ".mp4", ".mp3", ".ogg", ".m4a"}





def load_esc50_manifest(esc50_dir: pathlib.Path) -> Dict[str, str]:

    """Reads esc50.csv metadata and builds a filename-to-category mapping."""

    esc50_map = {}

    csv_candidates = list(esc50_dir.rglob("esc50.csv"))

    

    if not csv_candidates:

        print(f"Warning: esc50.csv not found in {esc50_dir}. Skipping ESC-50 metadata lookup.")

        return esc50_map



    csv_path = csv_candidates[0]

    with open(csv_path, mode="r", encoding="utf-8") as fh:

        reader = csv.DictReader(fh)

        for row in reader:

            filename = row["filename"]

            raw_category = row["category"].lower()

            mapped_category = ESC50_CATEGORY_MAP.get(raw_category, "unknown")

            esc50_map[filename] = mapped_category



    return esc50_map





def categorize(file_path: pathlib.Path, esc50_map: Dict[str, str]) -> str:

    """Categorizes an audio file based on metadata lookup or keyword matching."""

    # 1. ESC-50 exact metadata match

    if file_path.name in esc50_map:

        return esc50_map[file_path.name]



    # 2. FreeNoise keyword fallback

    stem = file_path.stem.lower()

    for keyword, category in KEYWORD_CATEGORY_MAP.items():

        if keyword in stem:

            return category



    return "unknown"





def build_manifest(noise_dir: str, out_csv: str) -> int:

    base_dir = pathlib.Path(noise_dir)

    esc50_map = load_esc50_manifest(base_dir)



    rows = []

    for file_path in base_dir.rglob("*"):

        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:

            category = categorize(file_path, esc50_map)

            rows.append({"path": str(file_path), "category": category})



    out_path = pathlib.Path(out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:

        writer = csv.DictWriter(fh, fieldnames=["path", "category"])

        writer.writeheader()

        writer.writerows(rows)



    unknown = sum(1 for r in rows if r["category"] == "unknown")

    print(

        f"Wrote {len(rows)} noise clips to {out_csv} ({unknown} unlabeled — "

        f"rename files, update ESC50_CATEGORY_MAP, or extend KEYWORD_CATEGORY_MAP)."

    )

    return len(rows)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(

        "--noise_dir",

        required=True,

        help="Directory containing FreeNoise and ESC-50 dataset folders",

    )

    parser.add_argument("--out_csv", required=True, help="Output manifest CSV path")

    args = parser.parse_args()

    build_manifest(args.noise_dir, args.out_csv)