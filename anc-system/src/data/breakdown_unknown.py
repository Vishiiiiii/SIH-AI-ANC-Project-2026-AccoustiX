import csv
from collections import Counter
import pathlib

rows = list(csv.DictReader(open("data/manifests/noise_manifest.csv")))
unknown_rows = [r for r in rows if r["category"] == "unknown"]

# Group unknowns by which top-level dataset folder they came from
def source_of(path):
    parts = pathlib.Path(path).parts
    lowered = [p.lower() for p in parts]
    if "impactset" in [p.replace("-", "") for p in lowered]:
        # go one level deeper: which impact-set subfolder
        idx = next(i for i,p in enumerate(lowered) if p.replace("-", "") == "impactset")
        sub = parts[idx+2] if len(parts) > idx+2 else "?"  # train/test then category
        return f"impactset/{sub}"
    if "sesa" in lowered:
        return "sesa"
    if "esc50" in lowered:
        return "esc50"
    if "freenoise" in lowered:
        return "freenoise"
    return "other"

by_source = Counter(source_of(r["path"]) for r in unknown_rows)
print(f"Total unknown: {len(unknown_rows)}\n")
for src, count in by_source.most_common():
    print(f"  {src}: {count}")
