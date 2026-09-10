"""
NLMS adaptive filter — the "true ANC" case, where a reference signal
correlated with the noise (e.g. an accelerometer or a second mic mounted
near the engine/rotor) is available. It adaptively predicts and subtracts
the noise component from the primary (mic) signal.

If you don't have a reference channel, use filters/wiener.py instead.
"""
import numpy as np


class NLMSFilter:
    def __init__(self, num_taps: int = 128, mu: float = 0.5, eps: float = 1e-6):
        self.num_taps = num_taps
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(num_taps, dtype=np.float64)

    def reset(self):
        self.weights[:] = 0.0

    def run(self, primary: np.ndarray, reference: np.ndarray):
        """
        primary:   the noisy microphone signal (speech + noise)
        reference: a signal correlated with the noise only (no speech)
        returns:   (error_signal, noise_estimate) — error_signal is the
                   noise-cancelled output.
        """
        n = len(primary)
        ref_padded = np.concatenate([np.zeros(self.num_taps - 1), reference])
        error = np.zeros(n, dtype=np.float64)
        noise_est = np.zeros(n, dtype=np.float64)

        for i in range(n):
            x = ref_padded[i:i + self.num_taps][::-1]
            y = float(np.dot(self.weights, x))
            e = primary[i] - y
            norm = float(np.dot(x, x)) + self.eps
            self.weights += (self.mu / norm) * e * x
            error[i] = e
            noise_est[i] = y

        return error.astype(np.float32), noise_est.astype(np.float32)
