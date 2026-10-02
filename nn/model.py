"""A minimal Sequential model with save/load support."""
import json

import numpy as np

from .layers import Dense, Dropout, ReLU, SoftmaxCrossEntropy, softmax


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train_step(self, x, y, optimizer):
        logits = self.forward(x, training=True)
        loss = self.loss_fn.forward(logits, y)
        self.backward(self.loss_fn.backward())
        optimizer.step()
        return loss, logits

    def predict_proba(self, x, batch_size=1000):
        out = [softmax(self.forward(x[i:i + batch_size], training=False))
               for i in range(0, len(x), batch_size)]
        return np.concatenate(out)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def evaluate(self, x, y, batch_size=1000):
        losses, correct = [], 0
        for i in range(0, len(x), batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            logits = self.forward(xb, training=False)
            losses.append(self.loss_fn.forward(logits, yb) * len(xb))
            correct += (logits.argmax(axis=1) == yb).sum()
        return sum(losses) / len(x), correct / len(x)

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        arch, arrays = [], {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                arch.append({"type": "Dense", "shape": list(layer.params["W"].shape)})
                arrays[f"{i}_W"] = layer.params["W"]
                arrays[f"{i}_b"] = layer.params["b"]
            elif isinstance(layer, ReLU):
                arch.append({"type": "ReLU"})
            elif isinstance(layer, Dropout):
                arch.append({"type": "Dropout", "p": layer.p})
        np.savez_compressed(path, architecture=json.dumps(arch), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        layers = []
        for i, spec in enumerate(json.loads(str(data["architecture"]))):
            if spec["type"] == "Dense":
                layer = Dense(*spec["shape"])
                layer.params["W"] = data[f"{i}_W"]
                layer.params["b"] = data[f"{i}_b"]
            elif spec["type"] == "ReLU":
                layer = ReLU()
            else:
                layer = Dropout(spec["p"])
            layers.append(layer)
        return cls(layers)


def build_mlp(sizes=(784, 256, 128, 10), dropout=0.2, seed=42):
    """Dense -> ReLU -> Dropout blocks, with a plain Dense output layer."""
    rng = np.random.default_rng(seed)
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Dense(a, b, rng))
        if i < len(sizes) - 2:
            layers.append(ReLU())
            if dropout > 0:
                layers.append(Dropout(dropout, rng))
    return Sequential(layers)
