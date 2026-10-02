"""Optimisers that update layer parameters in place from stored gradients."""
import numpy as np


class SGD:
    """Stochastic gradient descent with momentum."""

    def __init__(self, layers, lr=0.1, momentum=0.9, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}

    def step(self):
        for i, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                g = layer.grads[name] + self.weight_decay * p
                v = self.velocity.get((i, name), np.zeros_like(p))
                v = self.momentum * v - self.lr * g
                self.velocity[(i, name)] = v
                p += v


class Adam:
    """Adam: per-parameter adaptive learning rates (Kingma & Ba, 2015)."""

    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.m, self.v = {}, {}
        self.t = 0

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                key = (i, name)
                g = layer.grads[name] + self.weight_decay * p
                m = self.beta1 * self.m.get(key, 0.0) + (1 - self.beta1) * g
                v = self.beta2 * self.v.get(key, 0.0) + (1 - self.beta2) * g * g
                self.m[key], self.v[key] = m, v
                # bias correction: early steps would otherwise be too small
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
