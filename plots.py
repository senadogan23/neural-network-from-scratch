"""Figures used in the README. Saved as PNG with a white background."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#2563eb"
SECOND = "#f97316"
GOOD, BAD = "#16a34a", "#dc2626"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})


def training_curves(history, path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(epochs, history["train_loss"], color=ACCENT, marker="o", ms=3, label="train")
    a1.plot(epochs, history["val_loss"], color=SECOND, marker="o", ms=3, label="validation")
    a1.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    a2.plot(epochs, np.array(history["train_acc"]) * 100, color=ACCENT, marker="o", ms=3, label="train")
    a2.plot(epochs, np.array(history["val_acc"]) * 100, color=SECOND, marker="o", ms=3, label="validation")
    a2.set(title="Accuracy", xlabel="epoch", ylabel="%")
    for a in (a1, a2):
        a.grid(alpha=0.25)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_matrix(y_true, y_pred, path):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "#333")
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true",
           title=f"Confusion matrix (test, n={len(y_true):,})")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row share")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _grid(images, titles, colors, path, suptitle, cols=10):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.25))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, t, c in zip(np.ravel(axes), images, titles, colors):
        ax.imshow(img.reshape(28, 28), cmap="gray_r")
        ax.set_title(t, fontsize=9, color=c)
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sample_predictions(x, y, y_pred, path, n=20, seed=0):
    idx = np.random.default_rng(seed).choice(len(x), n, replace=False)
    _grid(x[idx], [f"pred {p}" for p in y_pred[idx]],
          [GOOD if p == t else BAD for p, t in zip(y_pred[idx], y[idx])],
          path, "Random test predictions")


def misclassified(x, y, y_pred, path, n=20):
    idx = np.nonzero(y != y_pred)[0][:n]
    _grid(x[idx], [f"{t} → {p}" for t, p in zip(y[idx], y_pred[idx])],
          [BAD] * len(idx), path, "Mistakes (true → predicted)")


def first_layer_weights(model, path, n=32):
    """Each hidden neuron's 784 input weights, reshaped back to 28x28.

    These are the 'templates' the first layer learned to detect.
    """
    W = model.layers[0].params["W"]
    fig, axes = plt.subplots(4, n // 4, figsize=(n // 4 * 1.0, 4.2))
    for k, ax in enumerate(np.ravel(axes)):
        w = W[:, k].reshape(28, 28)
        lim = np.abs(w).max()
        ax.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.axis("off")
    fig.suptitle("What the first layer learned (32 of the hidden neurons)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def prediction_bar(img28, probs, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.8),
                                 gridspec_kw={"width_ratios": [1, 2]})
    a1.imshow(img28, cmap="gray_r")
    a1.set_title("model input (28×28)")
    a1.axis("off")
    colors = [ACCENT if i == probs.argmax() else "#cbd5e1" for i in range(10)]
    a2.bar(range(10), probs * 100, color=colors)
    a2.set(xticks=range(10), ylim=(0, 100), ylabel="%",
           title=f"prediction: {probs.argmax()}  ({probs.max() * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
