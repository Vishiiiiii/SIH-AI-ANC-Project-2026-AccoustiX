"""
Export a trained checkpoint to ONNX for embedded deployment, optionally
INT8-quantized for cheap Raspberry Pi CPU inference.

Usage:
    python -m src.export_onnx --config configs/default.yaml --checkpoint checkpoints/best_model.pt --out model.onnx --quantize
"""
import argparse
import pathlib

import torch
import yaml

from src.models.rnnoise import RNNoiseStyle
from src.models.conv_tasnet import ConvTasNetLite
from src.models.dtln import DTLN
from src.realtime_demo import BLOCK_SECONDS, OVERLAP_SECONDS

MODEL_REGISTRY = {"rnnoise": RNNoiseStyle, "conv_tasnet": ConvTasNetLite, "dtln": DTLN}


def export(cfg, checkpoint_path, out_path, export_seconds=None, quantize=False):
    """
    export_seconds: fixed input length (in seconds) the exported model will
    accept. Defaults to BLOCK_SECONDS + OVERLAP_SECONDS from realtime_demo.py
    so the exported graph matches exactly what the demo feeds it per call
    (a lookback-context chunk, not just the new block — see ANCPipeline).
    STATIC shape on purpose: torch.stft/istft's shape inference doesn't
    reliably export with a dynamic time axis, and a fixed block size is
    realistic for streaming embedded inference anyway.

    quantize: also write a dynamically INT8-quantized copy alongside the
    FP32 export (`<out>.int8.onnx`) via onnxruntime's quantize_dynamic —
    no calibration dataset needed, and it's the right default for CPU/ARM
    inference on a Raspberry Pi (smaller file, faster matmuls, small
    accuracy cost). Re-run evaluate.py-style checks on the quantized model
    before trusting it for the final demo.
    """
    name = cfg["model"]
    params = cfg.get("model_params", {}).get(name, {})
    model = MODEL_REGISTRY[name](**params)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sample_rate = cfg["sample_rate"]
    export_seconds = export_seconds or (BLOCK_SECONDS + OVERLAP_SECONDS)
    dummy = torch.randn(1, int(sample_rate * export_seconds))

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["noisy_waveform"], output_names=["enhanced_waveform"],
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
    onnx_out = sess.run(None, {"noisy_waveform": dummy.numpy()})[0]
    max_diff = abs(torch_out - onnx_out).max()
    print(f"Max abs diff between PyTorch and ONNX output: {max_diff:.2e} "
          f"({'OK' if max_diff < 1e-3 else 'WARNING: check export'})")

    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        out_p = pathlib.Path(out_path)
        quant_path = str(out_p.with_suffix("")) + ".int8.onnx"
        quantize_dynamic(out_path, quant_path, weight_type=QuantType.QInt8)

        fp32_kb = out_p.stat().st_size / 1024
        int8_kb = pathlib.Path(quant_path).stat().st_size / 1024
        print(f"Quantized: {out_path} ({fp32_kb:.0f} KB) -> {quant_path} "
              f"({int8_kb:.0f} KB, {int8_kb / fp32_kb:.0%} of FP32 size)")

        # Parity check for the quantized model too — INT8 will legitimately
        # differ more than the FP32 check above; this just catches a broken
        # export, not judging audio quality (do that with evaluate.py).
        quant_sess = ort.InferenceSession(quant_path)
        quant_out = quant_sess.run(None, {"noisy_waveform": dummy.numpy()})[0]
        quant_diff = abs(torch_out - quant_out).max()
        print(f"Max abs diff between PyTorch (FP32) and quantized ONNX output: {quant_diff:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default=None,
                         help="Override the config's model: field to match the checkpoint being exported")
    parser.add_argument("--export_seconds", type=float, default=None,
                         help="Fixed input length in seconds the exported model expects "
                              "(default BLOCK_SECONDS + OVERLAP_SECONDS from realtime_demo.py)")
    parser.add_argument("--quantize", action="store_true",
                         help="Also write an INT8 dynamically-quantized <out>.int8.onnx, "
                              "recommended for Raspberry Pi CPU inference")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.model:
        cfg["model"] = args.model
    export(cfg, args.checkpoint, args.out, args.export_seconds, args.quantize)
