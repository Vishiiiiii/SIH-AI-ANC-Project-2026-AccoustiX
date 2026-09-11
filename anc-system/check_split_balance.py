import csv, collections
for split in ["train", "val", "test"]:
    rows = list(csv.DictReader(open(f"data/manifests/{split}.csv")))
    print(split, collections.Counter(r["category"] for r in rows))
