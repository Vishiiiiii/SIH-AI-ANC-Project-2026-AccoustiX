"""
Times NLMSFilter.run() on a block the same size realtime_demo.py actually
feeds it, so you know its real per-block cost BEFORE wiring it into the
real-time path. Run this on whatever hardware you'll actually deploy on
(your dev PC first, then the Raspberry Pi if you have one) -- the numbers
will likely differ a lot between the two.

Usage:
    python benchmark_nlms.py
"""
import time
import numpy as np

from src.filters.nlms import NLMSFilter

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.25
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)   # matches realtime_demo.py's BLOCK_SIZE

N_TRIALS = 50

for num_taps in [32, 64, 128, 256]:
    f = NLMSFilter(num_taps=num_taps, mu=0.5)
    primary = np.random.randn(BLOCK_SIZE).astype(np.float32)
    reference = np.random.randn(BLOCK_SIZE).astype(np.float32)

    # warm up (first call can include e.g. import/cache overhead)
    f.run(primary, reference)

    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        f.run(primary, reference)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    budget_ms = BLOCK_SECONDS * 1000
    print(f"num_taps={num_taps:4d}: mean={times.mean():7.2f}ms  p95={np.percentile(times, 95):7.2f}ms  "
          f"(block budget is {budget_ms:.0f}ms -- "
          f"{'OK, well under budget' if times.mean() < budget_ms * 0.3 else 'WARNING: eating a large chunk of the real-time budget' if times.mean() < budget_ms else 'FAILS real-time -- exceeds the block budget itself'})")

print("\nIf num_taps=128 (nlms.py's default) is at or near the block budget on your "
      "target hardware, don't wire it into realtime_demo.py as-is -- it needs a "
      "vectorized/block-LMS rewrite first (the current run() is a pure-Python "
      "per-sample loop, which is the likely bottleneck on the Pi's ARM CPU).")
