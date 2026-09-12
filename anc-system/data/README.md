# Local data is not versioned

Audio corpora, generated noisy/clean/reference WAV files, and CSV manifests
are intentionally excluded from Git. They are large, may have corpus license
restrictions, and the generated manifests contain absolute paths for the
machine that created them.

Use the scripts in `src/data/` and the workflow in `../DUALMIC_V1.md` to
recreate the data on another machine. Trained checkpoints and configurations
are versioned separately under `../checkpoints/` and `../configs/`.
