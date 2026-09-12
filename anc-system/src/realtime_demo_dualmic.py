"""Low-latency dual-microphone denoising demo.

Primary input (channel 0) contains speech plus noise. Reference input
(channel 1) should contain the same noise with as little speech as possible.
The pipeline is NLMS -> optional Wiener -> ONNX denoiser.

The ONNX graph determines the callback size. For example,
``acoustix_rnnoise_128ms.onnx`` uses 2,048 samples at 16 kHz, so its
algorithmic buffering delay is 128 ms instead of the old 250 ms.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd

from src.filters.nlms import NLMSFilter
from src.filters.wiener import WienerFilter


class DualMicANCPipeline:
    """Stateful block processor. One instance must serve the whole stream."""

    def __init__(self, onnx_path: str, sample_rate: int = 16000,
                 use_wiener: bool = True, use_nlms: bool = True,
                 nlms_taps: int = 128, nlms_mu: float = 0.002):
        options = ort.SessionOptions()
        # Small audio blocks do not benefit from a large thread pool; keeping
        # this single-threaded avoids callback jitter on lower-power hardware.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(onnx_path, sess_options=options)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.block_size = self._static_block_size(model_input.shape)
        self.model_channels = model_input.shape[1] if len(model_input.shape) == 3 else 1
        if self.model_channels not in (1, 2, 3):
            raise ValueError(f"Expected one, two, or three model input channels; received {model_input.shape!r}.")
        self.sample_rate = sample_rate
        self.nlms = NLMSFilter(num_taps=nlms_taps, mu=nlms_mu) if use_nlms else None
        self.wiener = WienerFilter(noise_estimate_ms=50) if use_wiener else None
        self.latencies: list[float] = []
        self.stage_latencies = {"nlms": [], "wiener": [], "model": []}

    @staticmethod
    def _static_block_size(shape) -> int:
        if len(shape) != 2 or not isinstance(shape[-1], int) or shape[-1] <= 0:
            raise ValueError(
                "This demo needs an ONNX model with a fixed [1, samples] input shape; "
                f"received {shape!r}."
            )
        return shape[-1]

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        primary = np.asarray(primary, dtype=np.float32).reshape(-1)
        reference = np.asarray(reference, dtype=np.float32).reshape(-1)
        if len(primary) != self.block_size or len(reference) != self.block_size:
            raise ValueError(
                f"Expected exactly {self.block_size} samples per channel, got "
                f"{len(primary)} and {len(reference)}."
            )

        total_start = time.perf_counter()
        enhanced_input = primary
        noise_estimate = reference
        if self.nlms is not None:
            start = time.perf_counter()
            enhanced_input, noise_estimate = self.nlms.run(enhanced_input, reference)
            self.stage_latencies["nlms"].append(time.perf_counter() - start)

        if self.wiener is not None:
            start = time.perf_counter()
            # NLMS maps the reference mic into the primary mic's acoustic
            # path. Its estimated noise is therefore a better Wiener estimate
            # than the raw reference (whose gain and delay differ).
            enhanced_input = self.wiener.process(
                enhanced_input, self.sample_rate, noise_reference=noise_estimate,
            )
            self.stage_latencies["wiener"].append(time.perf_counter() - start)

        start = time.perf_counter()
        if self.model_channels == 3:
            model_input = np.stack((enhanced_input, noise_estimate, primary))[None, :, :]
        elif self.model_channels == 2:
            model_input = np.stack((enhanced_input, noise_estimate))[None, :, :]
        else:
            model_input = enhanced_input[None, :]
        enhanced = self.session.run(None, {self.input_name: model_input})[0][0]
        self.stage_latencies["model"].append(time.perf_counter() - start)
        self.latencies.append(time.perf_counter() - total_start)
        return np.asarray(enhanced, dtype=np.float32)

    def timing_summary(self) -> str:
        if not self.latencies:
            return "No blocks processed."
        total = np.asarray(self.latencies) * 1000
        block_ms = self.block_size / self.sample_rate * 1000
        lines = [
            f"Processed {len(total)} blocks of {block_ms:.0f} ms.",
            f"Compute: mean {total.mean():.1f} ms, p95 {np.percentile(total, 95):.1f} ms, "
            f"RTF {total.mean() / block_ms:.2f}.",
            "Acoustic latency also includes one block plus the audio driver's input/output buffers.",
        ]
        for name, times in self.stage_latencies.items():
            if times:
                values = np.asarray(times) * 1000
                lines.append(f"  {name}: mean {values.mean():.1f} ms, p95 {np.percentile(values, 95):.1f} ms")
        return "\n".join(lines)


def run(onnx_path: str, input_device: int | None, output_device: int | None,
        sample_rate: int, use_wiener: bool, use_nlms: bool, nlms_taps: int,
        nlms_mu: float) -> None:
    pipeline = DualMicANCPipeline(
        onnx_path, sample_rate=sample_rate, use_wiener=use_wiener,
        use_nlms=use_nlms, nlms_taps=nlms_taps, nlms_mu=nlms_mu,
    )
    block_ms = pipeline.block_size / sample_rate * 1000

    def callback(indata, outdata, frames, _time_info, status):
        if status:
            print(f"Audio status: {status}")
        if frames != pipeline.block_size:
            raise sd.CallbackAbort(
                f"Audio callback supplied {frames} samples; ONNX model requires {pipeline.block_size}."
            )
        outdata[:, 0] = pipeline.process(indata[:, 0], indata[:, 1])

    print(
        f"Starting dual-mic pipeline: {pipeline.block_size} samples ({block_ms:.0f} ms), "
        f"NLMS={'on' if use_nlms else 'off'}, Wiener={'on' if use_wiener else 'off'}.\n"
        "Use headphones during testing to prevent speaker feedback. Press Ctrl+C to stop."
    )
    try:
        with sd.Stream(
            samplerate=sample_rate, blocksize=pipeline.block_size,
            channels=(2, 1), dtype="float32", callback=callback,
            device=(input_device, output_device), latency="low",
        ):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + pipeline.timing_summary())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a fixed-block ONNX model")
    parser.add_argument("--input-device", type=int, default=None,
                        help="Input device index from list_audio_devices.py")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Output device index from list_audio_devices.py")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--no-wiener", action="store_true", help="Skip the optional Wiener stage")
    parser.add_argument("--no-nlms", action="store_true", help="Skip the NLMS stage")
    parser.add_argument("--nlms-taps", type=int, default=128)
    parser.add_argument("--nlms-mu", type=float, default=0.002,
                        help="NLMS step size; use the value used to create the training data")
    args = parser.parse_args()
    run(
        args.model, args.input_device, args.output_device, args.sample_rate,
        use_wiener=not args.no_wiener, use_nlms=not args.no_nlms,
        nlms_taps=args.nlms_taps, nlms_mu=args.nlms_mu,
    )


if __name__ == "__main__":
    main()
