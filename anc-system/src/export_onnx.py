"""
Export a trained checkpoint to ONNX for embedded deployment.
INT8 quantization is a separate follow-up step (see README) once you've
confirmed parity between the PyTorch and exported-ONNX outputs.

Usage:
    python -m src.export_onnx --config configs/default.yaml --checkpoint checkpoints/best_model.pt --out model.onnx
"""
import argparse

import torch
import yaml

from src.models.rnnoise import RNNoiseStyle
from src.models.conv_tasnet import ConvTasNetLite
from src.models.dtln import DTLN
from src.models.dual_mic_rnnoise import DualMicRNNoiseStyle
from src.models.dual_mic_complex_rnnoise import DualMicComplexRNNoiseStyle

MODEL_REGISTRY = {"rnnoise": RNNoiseStyle, "conv_tasnet": ConvTasNetLite, "dtln": DTLN,
                  "dual_mic_rnnoise": DualMicRNNoiseStyle}
MODEL_REGISTRY["dual_mic_complex_rnnoise"] = DualMicComplexRNNoiseStyle


def export(cfg, checkpoint_path, out_path, export_seconds=None):
    """
    export_seconds: fixed block length (in seconds) the exported model will
    accept. Defaults to BLOCK_SECONDS in realtime_demo.py (0.25s) so the
    exported graph matches what the demo actually feeds it. STATIC shape on
    purpose: torch.stft/istft's shape inference doesn't reliably export with
    a dynamic time axis, and a fixed block size is realistic for streaming
    embedded inference anyway (see realtime_demo.py for the block-processing
    loop this shape is meant to match).
    """
    name = cfg["model"]
    params = cfg.get("model_params", {}).get(name, {})
    model = MODEL_REGISTRY[name](**params)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sample_rate = cfg["sample_rate"]
    export_seconds = export_seconds or 0.25
    samples = int(sample_rate * export_seconds)
    input_channels = getattr(model, "input_channels", 1)
    dummy = torch.randn(1, input_channels, samples) if input_channels > 1 else torch.randn(1, samples)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["dual_mic_waveforms" if input_channels > 1 else "noisy_waveform"], output_names=["enhanced_waveform"],
        opset_version=17,
        dynamo=False,  # torch>=2.7's new dynamo-based exporter needs the extra
                        # 'onnxscript' package; the legacy TorchScript-based
                        # exporter (dynamo=False) works out of the box and is
                        # plenty for this model size.
    )
    print(f"Exported to {out_path}")

    # Parity check
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(out_path)
    # Dual-mic graphs use ``dual_mic_waveforms`` while legacy models use
    # ``noisy_waveform``. Read the exported graph's actual input name rather
    # than assuming the legacy one, so parity validates every model variant.
    onnx_input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {onnx_input_name: dummy.numpy()})[0]
    max_diff = abs(torch_out - onnx_out).max()
    print(f"Max abs diff between PyTorch and ONNX output: {max_diff:.2e} "
          f"({'OK' if max_diff < 1e-3 else 'WARNING: check export'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--export_seconds", type=float, default=None,
                         help="Fixed block length in seconds the exported model expects "
                              "(default 0.25s, matching realtime_demo.py's BLOCK_SECONDS)")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    export(cfg, args.checkpoint, args.out, args.export_seconds)
