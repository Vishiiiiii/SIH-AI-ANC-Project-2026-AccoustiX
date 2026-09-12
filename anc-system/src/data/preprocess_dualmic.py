"""Precompute the classical dual-mic front end for model training.

Input rows must contain ``noisy_path``, ``clean_path`` and ``reference_path``.
The resulting manifest keeps the original metadata, stores ``raw_noisy_path``
and ``nlms_path``, and makes ``noisy_path`` point to the exact NLMS/Wiener
signal that the neural model will receive at runtime.

Example:
    python -m src.data.preprocess_dualmic \
      --input_manifest data/dual_mic_v1/manifests/train.csv \
      --out_dir data/dual_mic_v1/front_end/train \
      --output_manifest data/dual_mic_v1/manifests_front_end/train.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from src.filters.nlms import NLMSFilter
from src.filters.wiener import WienerFilter


def accelerated_nlms_runner(num_taps: int, mu: float, eps: float = 1e-6):
    """Return a Numba-compiled NLMS function for the offline dataset build.

    This is intentionally separate from ``NLMSFilter``: real-time deployment
    must not spend seconds JIT-compiling inside an audio callback, whereas the
    one-time dataset build benefits substantially from native loops.
    """
    try:
        from numba import njit
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "--accelerated needs the optional numba package. Run without it "
            "to use the portable NumPy implementation."
        ) from exc

    @njit
    def run(primary, reference):
        weights = np.zeros(num_taps, dtype=np.float32)
        history = np.zeros(num_taps - 1, dtype=np.float32)
        padded = np.empty(len(reference) + num_taps - 1, dtype=np.float32)
        padded[:num_taps - 1] = history
        padded[num_taps - 1:] = reference
        error = np.empty(len(primary), dtype=np.float32)
        noise_estimate = np.empty(len(primary), dtype=np.float32)
        for i in range(len(primary)):
            estimate = np.float32(0.0)
            norm = np.float32(eps)
            for tap in range(num_taps):
                value = padded[i + num_taps - 1 - tap]
                estimate += weights[tap] * value
                norm += value * value
            residual = primary[i] - estimate
            update = mu * residual / norm
            for tap in range(num_taps):
                weights[tap] += update * padded[i + num_taps - 1 - tap]
            error[i] = residual
            noise_estimate[i] = estimate
        return error, noise_estimate

    return run


def preprocess_manifest(input_manifest: str, out_dir: str, output_manifest: str,
                        sample_rate: int = 16000, nlms_taps: int = 128,
                        nlms_mu: float = 0.002, use_wiener: bool = True,
                        accelerated: bool = False, block_samples: int | None = None) -> None:
    with open(input_manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input manifest is empty: {input_manifest}")
    required = {"noisy_path", "clean_path", "reference_path"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"{input_manifest} lacks {sorted(missing)}. Rebuild it with --with_reference."
        )

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fast_nlms = accelerated_nlms_runner(nlms_taps, nlms_mu) if accelerated else None
    processed_rows = []
    for index, row in enumerate(tqdm(rows, desc=f"front end: {Path(input_manifest).stem}")):
        primary, primary_sr = sf.read(row["noisy_path"], dtype="float32")
        reference, reference_sr = sf.read(row["reference_path"], dtype="float32")
        if primary_sr != sample_rate or reference_sr != sample_rate:
            raise ValueError(
                f"Expected {sample_rate} Hz; got {row['noisy_path']}={primary_sr}, "
                f"{row['reference_path']}={reference_sr}."
            )
        # Audio files are independent training examples, so filter state must
        # not leak from one row into the next. Within each clip NLMS is fully
        # continuous, including the reference delay line.
        if fast_nlms is not None:
            nlms_audio, noise_estimate = fast_nlms(primary, reference)
        else:
            nlms = NLMSFilter(num_taps=nlms_taps, mu=nlms_mu)
            nlms_audio, noise_estimate = nlms.run(primary, reference)

        if not use_wiener:
            front_end_audio = nlms_audio
        elif block_samples is None:
            # Offline analysis mode: the classical filter sees the full clip.
            front_end_audio = WienerFilter(noise_estimate_ms=50).process(
                nlms_audio, sample_rate, noise_reference=noise_estimate,
            )
        else:
            # Deployment-faithful mode: ``realtime_demo_dualmic.py`` calls
            # WienerFilter.process once per audio callback.  The filter is
            # intentionally stateless, so doing the same here avoids training
            # on a more favourable whole-clip front end that cannot exist live.
            if block_samples < 512:
                raise ValueError("block_samples must be at least the Wiener FFT size (512).")
            blocks = []
            for start in range(0, len(nlms_audio), block_samples):
                stop = start + block_samples
                blocks.append(WienerFilter(noise_estimate_ms=50).process(
                    nlms_audio[start:stop], sample_rate,
                    noise_reference=noise_estimate[start:stop],
                ))
            import numpy as np
            front_end_audio = np.concatenate(blocks).astype("float32", copy=False)
        nlms_path = destination / f"nlms_{index:05d}.wav"
        output_path = destination / f"front_end_{index:05d}.wav"
        sf.write(nlms_path, nlms_audio, sample_rate)
        sf.write(output_path, front_end_audio, sample_rate)

        output_row = dict(row)
        output_row["raw_noisy_path"] = row["noisy_path"]
        output_row["nlms_path"] = str(nlms_path)
        output_row["noisy_path"] = str(output_path)
        output_row["front_end"] = "nlms+wiener" if use_wiener else "nlms"
        output_row["front_end_block_samples"] = "whole_clip" if block_samples is None else str(block_samples)
        processed_rows.append(output_row)

    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(processed_rows[0]))
        writer.writeheader()
        writer.writerows(processed_rows)
    print(f"Wrote {len(processed_rows)} processed rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--nlms_taps", type=int, default=128)
    parser.add_argument("--nlms_mu", type=float, default=0.002,
                        help="NLMS step size; calibrated on held-out synthetic pairs")
    parser.add_argument("--no_wiener", action="store_true")
    parser.add_argument("--accelerated", action="store_true",
                        help="Use Numba to accelerate this one-time dataset build")
    parser.add_argument("--block_samples", type=int, default=None,
                        help="Process Wiener in fixed callback-size blocks. Set 2048 to exactly match the live 128 ms path.")
    args = parser.parse_args()
    preprocess_manifest(
        args.input_manifest, args.out_dir, args.output_manifest,
        sample_rate=args.sample_rate, nlms_taps=args.nlms_taps,
        nlms_mu=args.nlms_mu, use_wiener=not args.no_wiener,
        accelerated=args.accelerated, block_samples=args.block_samples,
    )


if __name__ == "__main__":
    main()
