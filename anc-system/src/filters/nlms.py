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
        if num_taps < 1:
            raise ValueError("num_taps must be at least 1")
        self.weights = np.zeros(num_taps, dtype=np.float32)
        # Keep the tail of the reference channel between audio callbacks.
        # Resetting this history every block creates a short transient at each
        # boundary and prevents the adaptive filter from behaving as a single
        # continuous filter.
        self.reference_history = np.zeros(num_taps - 1, dtype=np.float32)

    def reset(self):
        self.weights[:] = 0.0
        self.reference_history[:] = 0.0

    def run(self, primary: np.ndarray, reference: np.ndarray):
        """
        primary:   the noisy microphone signal (speech + noise)
        reference: a signal correlated with the noise only (no speech)
        returns:   (error_signal, noise_estimate) — error_signal is the
                   noise-cancelled output.
        """
        primary = np.asarray(primary, dtype=np.float32).reshape(-1)
        reference = np.asarray(reference, dtype=np.float32).reshape(-1)
        if len(primary) != len(reference):
            raise ValueError("primary and reference must have the same number of samples")

        n = len(primary)
        ref_padded = np.concatenate([self.reference_history, reference])
        error = np.empty(n, dtype=np.float32)
        noise_est = np.empty(n, dtype=np.float32)

        for i in range(n):
            x = ref_padded[i:i + self.num_taps][::-1]
            y = np.dot(self.weights, x)
            e = primary[i] - y
            norm = np.dot(x, x) + self.eps
            self.weights += (self.mu / norm) * e * x
            error[i] = e
            noise_est[i] = y

        if self.num_taps > 1:
            self.reference_history[:] = ref_padded[-(self.num_taps - 1):]
        return error, noise_est
