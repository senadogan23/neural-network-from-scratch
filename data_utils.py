"""MNIST download/loading and image preprocessing utilities."""
import os
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_mnist(path=None):
    """Return (x_train, y_train), (x_test, y_test) with pixels scaled to [0, 1].

    Images are flattened to 784-dim vectors. The file is downloaded once
    (~11 MB) and cached in ./data.
    """
    path = path or os.path.join(DATA_DIR, "mnist.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading MNIST to {path} ...")
        urllib.request.urlretrieve(MNIST_URL, path)
    with np.load(path) as d:
        x_train, y_train = d["x_train"], d["y_train"]
        x_test, y_test = d["x_test"], d["y_test"]
    prep = lambda x: x.reshape(len(x), -1).astype(np.float32) / 255.0
    return (prep(x_train), y_train.astype(np.int64)), (prep(x_test), y_test.astype(np.int64))


def preprocess_image(path):
    """Turn a photo/drawing of a single digit into an MNIST-style 28x28 input.

    MNIST digits are white on black, fit in a 20x20 box and are centred by
    centre of mass inside a 28x28 frame. Reproducing those steps is what
    makes the model work on images it has never seen.
    """
    from PIL import Image

    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    # 1. white digit on black background: invert if the background is light
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    if border.mean() > 0.5:
        img = 1.0 - img

    # 2. remove background noise with a soft threshold
    img = np.clip((img - 0.2) / 0.8, 0.0, 1.0)

    # 3. crop to the bounding box of the digit
    ys, xs = np.nonzero(img > 0.1)
    if len(ys) == 0:
        raise ValueError("No digit found in the image.")
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 4. resize so the longest side is 20 px, keeping the aspect ratio
    h, w = img.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray((img * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS)
    small = np.asarray(small, dtype=np.float32) / 255.0

    # 5. paste into a 28x28 canvas, centred by centre of mass
    canvas = np.zeros((28, 28), dtype=np.float32)
    total = small.sum()
    cy = (small.sum(axis=1) * np.arange(new_h)).sum() / total
    cx = (small.sum(axis=0) * np.arange(new_w)).sum() / total
    top = int(np.clip(round(14 - cy), 0, 28 - new_h))
    left = int(np.clip(round(14 - cx), 0, 28 - new_w))
    canvas[top:top + new_h, left:left + new_w] = small
    return canvas
