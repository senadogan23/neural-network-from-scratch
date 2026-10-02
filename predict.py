"""Predict the digit in your own image (photo, scan or drawing).

    python predict.py my_digit.png
    python predict.py my_digit.png --show     # also saves a probability chart
"""
import argparse
import os

from data_utils import preprocess_image
from nn import Sequential


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="path to an image containing one digit")
    p.add_argument("--model", default="outputs/model.npz")
    p.add_argument("--show", action="store_true",
                   help="save <image>_prediction.png with the input and probabilities")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found at {args.model}. Run `python train.py` first.")

    model = Sequential.load(args.model)
    img = preprocess_image(args.image)
    probs = model.predict_proba(img.reshape(1, -1))[0]

    print(f"Prediction: {probs.argmax()}  (confidence {probs.max() * 100:.1f}%)")
    for digit in probs.argsort()[::-1][:3]:
        print(f"  {digit}: {probs[digit] * 100:5.1f}%")

    if args.show:
        import plots
        out = os.path.splitext(args.image)[0] + "_prediction.png"
        plots.prediction_bar(img, probs, out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
