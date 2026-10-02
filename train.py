"""Train the from-scratch MLP on MNIST and save the model + figures.

    python train.py                      # defaults: 784-256-128-10, Adam, 15 epochs
    python train.py --epochs 5 --optimizer sgd --lr 0.05
"""
import argparse
import json
import os
import time

import numpy as np

from data_utils import load_mnist
from nn import SGD, Adam, build_mlp
import plots


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 for Adam, 0.05 for SGD")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    # hold out part of the training set for validation; the test set is
    # only touched once, at the very end
    idx = rng.permutation(len(x_train))
    val_idx, tr_idx = idx[:args.val_size], idx[args.val_size:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    print(f"train {len(x_tr)} | val {len(x_val)} | test {len(x_test)}")

    model = build_mlp((784, *args.hidden, 10), dropout=args.dropout, seed=args.seed)
    lr = args.lr or (1e-3 if args.optimizer == "adam" else 0.05)
    opt = (Adam(model.layers, lr=lr) if args.optimizer == "adam"
           else SGD(model.layers, lr=lr, momentum=0.9))
    n_params = sum(p.size for l in model.layers for p in l.params.values())
    print(f"model {[784, *args.hidden, 10]} | {n_params:,} parameters | {args.optimizer} lr={lr}")

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc, best_path = 0.0, os.path.join(args.out, "model.npz")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        # simple step decay: halve the learning rate after 60% of training
        if epoch == int(args.epochs * 0.6) + 1:
            opt.lr /= 2

        order = rng.permutation(len(x_tr))
        losses, correct = [], 0
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            loss, logits = model.train_step(x_tr[b], y_tr[b], opt)
            losses.append(loss * len(b))
            correct += (logits.argmax(axis=1) == y_tr[b]).sum()

        tr_loss, tr_acc = sum(losses) / len(x_tr), correct / len(x_tr)
        val_loss, val_acc = model.evaluate(x_val, y_val)
        for k, v in zip(history, (tr_loss, tr_acc, val_loss, val_acc)):
            history[k].append(float(v))

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(best_path)
            marker = "  * saved"
        print(f"epoch {epoch:2d}/{args.epochs} | loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {time.time() - start:5.1f}s{marker}")

    # final evaluation with the best checkpoint
    from nn import Sequential
    model = Sequential.load(best_path)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%  (loss {test_loss:.4f})")

    y_pred = model.predict(x_test)
    plots.training_curves(history, os.path.join(args.out, "training_curves.png"))
    plots.confusion_matrix(y_test, y_pred, os.path.join(args.out, "confusion_matrix.png"))
    plots.sample_predictions(x_test, y_test, y_pred, os.path.join(args.out, "predictions.png"))
    plots.misclassified(x_test, y_test, y_pred, os.path.join(args.out, "misclassified.png"))
    plots.first_layer_weights(model, os.path.join(args.out, "weights.png"))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": float(test_acc), "test_loss": float(test_loss),
                   "best_val_accuracy": float(best_acc), "parameters": int(n_params),
                   "train_time_sec": round(time.time() - start, 1),
                   "args": vars(args), "history": history}, f, indent=2)
    print(f"Figures and model saved to ./{args.out}/")


if __name__ == "__main__":
    main()
