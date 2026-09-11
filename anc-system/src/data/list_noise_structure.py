import pathlib
 
def normalize(s):
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")
 
root_candidates = [pathlib.Path("data/raw/noise"), pathlib.Path.cwd() / "data" / "raw" / "noise"]
noise_root = next((p for p in root_candidates if p.exists()), None)
 
if noise_root is None:
    print(f"Could not find data/raw/noise from current directory: {pathlib.Path.cwd()}")
    print("Run this from the project root (the folder that directly contains 'data\\').")
    raise SystemExit(1)
 
print(f"Using noise root: {noise_root.resolve()}\n")
 
for target in ["impactset", "sesa"]:
    matches = [d for d in noise_root.rglob("*") if d.is_dir() and normalize(d.name) == target]
    if not matches:
        print(f"--- {target} --- NOT FOUND anywhere under {noise_root}")
        continue
    for d in matches:
        print(f"--- {d.relative_to(noise_root)} ---")
        for sub in ["train", "test"]:
            subdir = d / sub
            if subdir.exists():
                all_items = [f for f in subdir.rglob("*") if f.is_file()]
                print(f"  {sub}: {len(all_items)} total items, first 10:")
                for f in all_items[:10]:
                    print("   ", f.relative_to(d))
            else:
                print(f"  {sub}: NOT FOUND at {subdir}")
    print()
 
