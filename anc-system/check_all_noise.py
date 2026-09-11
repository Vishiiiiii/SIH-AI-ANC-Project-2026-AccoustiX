import soundfile as sf, pathlib, csv

rows = list(csv.DictReader(open("data/manifests/noise_manifest.csv")))
bad = []
for r in rows:
    try:
        sf.info(r["path"])
    except Exception as e:
        bad.append((r["path"], str(e)))

print(f"{len(bad)} unreadable files out of {len(rows)}")
for p, e in bad:
    print(" ", p)