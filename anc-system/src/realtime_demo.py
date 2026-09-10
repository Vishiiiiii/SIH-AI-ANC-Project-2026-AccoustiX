"""
Live microphone -> hybrid ANC pipeline -> speaker demo.

Usage:
    python -m src.realtime_demo --model model.onnx --classical wiener

IMPORTANT — this is a BLOCK-PROCESSING demo, not a fully causal streaming
one: each callback grabs a fixed-size chunk, runs the classical filter then
the ONNX model on that whole chunk, and plays it back. That's good enough
to prove the concept and measure real end-to-end latency, but the frequency-
domain models (rnnoise.py, dtln.py stage 1) technically need overlap-add
across chunk boundaries to avoid clicks at chunk edges — see the TODO below
before treating this as the final embedded implementation.
"""
import argparse
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd

from src.filters.wiener import wiener_denoise

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.25          # chunk size fed to the model each callback
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)


class ANCPipeline:
    def __init__(self, onnx_path: str, use_classical: bool = True):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.use_classical = use_classical
        self.latencies = []

    def process(self, block: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        x = block
        if self.use_classical:
            # Cheap first pass — see filters/wiener.py note: this re-estimates
            # noise from the first ms of EVERY block, which is a simplification;
            # for a real deployment, keep a running noise estimate across blocks.
            x = wiener_denoise(x, SAMPLE_RATE, noise_estimate_ms=50)

        onnx_in = x.astype(np.float32)[None, :]      # (1, T)
        enhanced = self.session.run(None, {self.input_name: onnx_in})[0][0]

        self.latencies.append(time.perf_counter() - t0)
        return enhanced.astype(np.float32)


def run(onnx_path: str, use_classical: bool):
    pipeline = ANCPipeline(onnx_path, use_classical)

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status)
        mono_in = indata[:, 0]
        outdata[:, 0] = pipeline.process(mono_in)

    print(f"Starting real-time demo — block size {BLOCK_SECONDS*1000:.0f} ms, "
          f"classical pre-filter {'ON' if use_classical else 'OFF'}. Ctrl+C to stop.")
    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1,
                    dtype="float32", callback=callback):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    if pipeline.latencies:
        lat_ms = np.array(pipeline.latencies) * 1000
        rtf = np.mean(lat_ms) / (BLOCK_SECONDS * 1000)
        print(f"\nProcessed {len(lat_ms)} blocks. "
              f"Mean latency: {lat_ms.mean():.1f} ms, p95: {np.percentile(lat_ms, 95):.1f} ms, "
              f"Real-time factor (RTF): {rtf:.2f} (target < 1.0, ideally < 0.3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to exported .onnx model")
    parser.add_argument("--no-classical", action="store_true",
                         help="Skip the classical Wiener pre-filter, run the DL model alone")
    args = parser.parse_args()
    run(args.model, use_classical=not args.no_classical)

# TODO before final submission:
# 1. Overlap-add between blocks for the frequency-domain models to remove
#    clicking at block boundaries (or switch to a streaming STFT with
#    persistent GRU/LSTM state — requires refactoring rnnoise.py / dtln.py
#    forward() to accept/return hidden state explicitly).
# 2. Maintain a running noise estimate for the classical filter instead of
#    re-estimating from the first 50ms of every 250ms block.
# 3. Log before/after audio clips automatically for the pre-recorded demo
#    fallback the plan's risk table calls for.
