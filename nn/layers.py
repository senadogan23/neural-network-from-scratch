"""Neural network layers implemented with NumPy only.

Every layer follows the same contract:
    forward(x, training)  -> output, caching whatever backward needs
    backward(grad_out)    -> gradient w.r.t. the layer input,
                             storing parameter gradients in self.grads
"""
import numpy as np


class Layer:
    params: dict
    grads: dict

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer: y = x @ W + b

    Weights use He initialisation, which keeps activation variance stable
    through ReLU layers.
    """

    def __init__(self, in_features, out_features, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng()
        std = np.sqrt(2.0 / in_features)
        self.params["W"] = rng.normal(0.0, std, (in_features, out_features))
        self.params["b"] = np.zeros(out_features)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):
        # dL/dW = x^T · dL/dy,  dL/db = sum over batch,  dL/dx = dL/dy · W^T
        self.grads["W"] = self.x.T @ grad_out
        self.grads["b"] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class Dropout(Layer):
    """Inverted dropout: active only during training, identity at inference."""

    def __init__(self, p=0.2, rng=None):
        super().__init__()
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x, training=True):
        if not training or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out if self.mask is None else grad_out * self.mask


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax + cross-entropy fused into one step.

    Combining them gives the very simple gradient (probs - one_hot) / N
    and avoids the numerical problems of differentiating softmax alone.
    """

    def forward(self, logits, labels):
        self.probs = softmax(logits)
        self.labels = labels
        n = logits.shape[0]
        correct = self.probs[np.arange(n), labels]
        return -np.mean(np.log(correct + 1e-12))

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.labels] -= 1.0
        return grad / n
