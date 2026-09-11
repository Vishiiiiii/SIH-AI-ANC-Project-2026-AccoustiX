import pathlib, re
from collections import Counter

root_candidates = [pathlib.Path("data/raw/noise"), pathlib.Path.cwd() / "data" / "raw" / "noise"]
noise_root = next((p for p in root_candidates if p.exists()), None)
if noise_root is None:
    print(f"Could not find data/raw/noise from {pathlib.Path.cwd()}")
    raise SystemExit(1)

def normalize(s):
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")

for target in ["impactset", "sesa"]:
    matches = [d for d in noise_root.rglob("*") if d.is_dir() and normalize(d.name) == target]
    for d in matches:
        print(f"=== {d.relative_to(noise_root)} ===")
        if target == "impactset":
            for sub in ["train", "test"]:
                subdir = d / sub
                if not subdir.exists():
                    continue
                subfolders = sorted(set(p.name for p in subdir.iterdir() if p.is_dir()))
                print(f"  {sub} subfolders: {subfolders}")
        else:  # sesa: filenames are flat, prefix before first "_" or digit run
            for sub in ["train", "test"]:
                subdir = d / sub
                if not subdir.exists():
                    continue
                prefixes = Counter()
                for f in subdir.rglob("*"):
                    if f.is_file():
                        m = re.match(r"^([a-zA-Z]+)", f.stem)
                        if m:
                            prefixes[m.group(1).lower()] += 1
                print(f"  {sub} filename prefixes: {dict(prefixes)}")
    print()
    