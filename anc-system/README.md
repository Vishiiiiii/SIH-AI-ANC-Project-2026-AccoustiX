# AcoustiX dual-microphone prototype

AcoustiX enhances speech from a primary microphone using a second microphone
as a noise reference:

`primary mic → NLMS → Wiener → neural refinement → headphones`

The live path operates on 2,048 samples at 16 kHz (128 ms). The network input
is shaped for the selected model: v2 uses two channels and v3 uses three.

## Included checkpoints

All checkpoint files are included in Git; local audio datasets are not.

| Checkpoint | Role | Held-out result |
| --- | --- | --- |
| `checkpoints/dualmic_v2_live/best_model.pt` | Recommended SNR-focused dual-mic model | SNR 12.98 dB, PESQ 2.016, STOI 0.910 |
| `checkpoints/dualmic_v3_complex/best_model.pt` | Complex-mask experiment, best PESQ | SNR 11.10 dB, PESQ 2.065, STOI 0.909 |
| `checkpoints/dualmic_v1/acoustix_dualmic_128ms.onnx` | Earlier two-channel ONNX reference | Exported and parity-checked |

The project target is SNR above 15 dB, STOI above 0.85, and PESQ above 2.5.
Neither v2 nor v3 reaches every target yet. V2 is the current deployment
baseline because it has the highest conventional SNR; v3 is retained as a
quality experiment because it improves PESQ.

## Setup

Use Python 3.13 with a CUDA-enabled PyTorch installation if GPU inference is
needed, then install dependencies:

```powershell
pip install -r requirements.txt
```

## Export a checkpoint to ONNX

Export v2 (two model input channels):

```powershell
python -m src.export_onnx --config configs/dualmic_v2_live.yaml --checkpoint checkpoints/dualmic_v2_live/best_model.pt --out checkpoints/dualmic_v2_live/acoustix_dualmic_v2_128ms.onnx --export_seconds 0.128
```

Export v3 (three model input channels):

```powershell
python -m src.export_onnx --config configs/dualmic_v3_complex.yaml --checkpoint checkpoints/dualmic_v3_complex/best_model.pt --out checkpoints/dualmic_v3_complex/acoustix_dualmic_v3_128ms.onnx --export_seconds 0.128
```

Each export performs a PyTorch-versus-ONNX numerical parity check.

## Test with two microphones

List available sound devices:

```powershell
python list_audio_devices.py
```

Then run the live v2 pipeline after exporting it. Replace the two device
numbers with the primary/reference input device and headphones output device:

```powershell
python -m src.realtime_demo_dualmic --model checkpoints/dualmic_v2_live/acoustix_dualmic_v2_128ms.onnx --input-device INPUT_DEVICE --output-device OUTPUT_DEVICE --nlms-mu 0.002
```

Use headphones: speakers can feed back into both microphones. The primary
input must expose two channels in this order: primary speech/noise mic first,
noise-reference mic second. The reference mic should receive as little target
speech as possible.

## Training and full evaluation

The audio datasets are excluded because they are large and the manifests
contain local paths. To train or run the 1,500-example evaluation, obtain the
corpora, rebuild the manifest/dataset pipeline, then run:

```powershell
python -m src.train --config configs/dualmic_v2_live.yaml
python -m src.evaluate --config configs/dualmic_v2_live.yaml --checkpoint checkpoints/dualmic_v2_live/best_model.pt
```

See [DUALMIC_V1.md](DUALMIC_V1.md) for the reproducible raw-data workflow.
