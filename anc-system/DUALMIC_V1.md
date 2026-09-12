# AcoustiX dual-mic v1 pipeline

This is a new, reproducible data flow built from raw audio. Existing `data/processed`,
old manifests and checkpoints are deliberately not used or overwritten.

## 1. Build the raw-noise manifest

```powershell
python -m src.data.noise_manifest --noise_dir data/raw/noise --out_csv data/dual_mic_v1/manifests/noise_manifest.csv
```

Inspect the unknown-category count. Add category mappings before continuing if it is large.

## 2. Build clean/noisy/reference triples

```powershell
python -m src.data.build_dataset --speech_dir data/raw/speech/LibriSpeech --noise_manifest data/dual_mic_v1/manifests/noise_manifest.csv --out_dir data/dual_mic_v1/raw_pairs --manifest_dir data/dual_mic_v1/manifests --with_reference --clip_seconds 4 --n_train 12000 --n_val 1500 --n_test 1500 --seed 2026
```

Every row contains `noisy_path`, `clean_path`, and a synthetic `reference_path`.
The reference is derived only from the selected noise track, with random gain, delay,
and self-noise. The delay is applied to the primary path, so the reference leads the
primary noise—the direction required by a causal NLMS filter. It is a stand-in until
captures from the two INMP441 microphones exist.
Sampling is balanced across stationary, non-stationary, and impulsive categories by
default; use `--natural_category_distribution` only when the source-file distribution
itself is desired. The raw noise sources are also split before pair generation, so a
single source recording never crosses from training into validation or test.

## 3. Apply the same front end used live

Run once for each split. This saves the NLMS residual and the optional Wiener residual;
the output `noisy_path` is what RNNoise trains on.

```powershell
python -m src.data.preprocess_dualmic --input_manifest data/dual_mic_v1/manifests/train.csv --out_dir data/dual_mic_v1/front_end/train --output_manifest data/dual_mic_v1/manifests_front_end/train.csv --nlms_mu 0.002 --accelerated
python -m src.data.preprocess_dualmic --input_manifest data/dual_mic_v1/manifests/val.csv --out_dir data/dual_mic_v1/front_end/val --output_manifest data/dual_mic_v1/manifests_front_end/val.csv --nlms_mu 0.002 --accelerated
python -m src.data.preprocess_dualmic --input_manifest data/dual_mic_v1/manifests/test.csv --out_dir data/dual_mic_v1/front_end/test --output_manifest data/dual_mic_v1/manifests_front_end/test.csv --nlms_mu 0.002 --accelerated
```

## 4. Train and export a 128-ms model

```powershell
python -m src.train --config configs/dualmic_v1.yaml
python -m src.export_onnx --config configs/dualmic_v1.yaml --checkpoint checkpoints/dualmic_v1/best_model.pt --out checkpoints/dualmic_v1/acoustix_dualmic_128ms.onnx --export_seconds 0.128
```

## 5. Use the model with the two microphones

```powershell
python -m src.realtime_demo_dualmic --model checkpoints/dualmic_v1/acoustix_dualmic_128ms.onnx --input-device INPUT_INDEX --output-device OUTPUT_INDEX
```

The two microphones must appear as one two-channel input device. The primary mic faces the speaker; the reference mic should be nearer the noise source and away from the speaker. Use headphones while testing to prevent acoustic feedback.
