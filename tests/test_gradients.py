"""Gradient check: compare analytic backprop gradients with finite differences.

If backpropagation is implemented correctly, the two should agree to ~1e-7.
Run with:  python -m pytest -q   (or simply: python tests/test_gradients.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn import build_mlp  # noqa: E402


def relative_error(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12)


def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    model = build_mlp((12, 8, 6, 4), dropout=0.0, seed=0)  # dropout off: deterministic
    x = rng.normal(size=(5, 12))
    y = rng.integers(0, 4, size=5)

    def loss():
        return model.loss_fn.forward(model.forward(x, training=False), y)

    loss()
    model.backward(model.loss_fn.backward())

    eps = 1e-6
    for layer in model.layers:
        for name, p in layer.params.items():
            analytic = layer.grads[name].copy()
            numeric = np.zeros_like(p)
            for i in np.ndindex(p.shape):
                old = p[i]
                p[i] = old + eps
                plus = loss()
                p[i] = old - eps
                minus = loss()
                p[i] = old
                numeric[i] = (plus - minus) / (2 * eps)
            err = relative_error(analytic, numeric)
            assert err < 1e-5, f"{type(layer).__name__}.{name}: relative error {err:.2e}"


def test_loss_decreases_on_tiny_batch():
    """A network should be able to memorise 20 random samples."""
    from nn import Adam

    rng = np.random.default_rng(1)
    model = build_mlp((10, 32, 3), dropout=0.0, seed=1)
    opt = Adam(model.layers, lr=1e-2)
    x, y = rng.normal(size=(20, 10)), rng.integers(0, 3, size=20)
    first = model.train_step(x, y, opt)[0]
    for _ in range(300):
        last = model.train_step(x, y, opt)[0]
    assert last < first * 0.05


if __name__ == "__main__":
    test_backprop_matches_numerical_gradient()
    test_loss_decreases_on_tiny_batch()
    print("All gradient checks passed.")
