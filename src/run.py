import argparse
from pathlib import Path

import pandas as pd

from .config import EPOCHS, LABELS, MODEL_NAME, SEEDS, TRAIN_SIZES
from .data import load_data, sample_per_class
from .models import scores, train_tfidf, train_transformer
from .plot import plot_learning_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Low-data GoEmotions experiment")
    parser.add_argument("--sizes", nargs="+", type=int, default=TRAIN_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-transformer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    rows, prediction_rows = [], []

    for size in args.sizes:
        for seed in args.seeds:
            train = sample_per_class(data["train"], size, seed)
            print(f"Training with {size} examples per class (seed {seed})")
            runs = [("tfidf", *train_tfidf(train, data["validation"], data["test"]))]
            if not args.skip_transformer:
                runs.append(
                    (
                        "distilbert",
                        *train_transformer(
                            train,
                            data["validation"],
                            data["test"],
                            args.model,
                            args.epochs,
                            seed,
                        ),
                    )
                )

            for model, seconds, predictions in runs:
                for split in ("validation", "test"):
                    rows.append(
                        {
                            "model": model,
                            "train_per_class": size,
                            "total_train": size * len(LABELS),
                            "seed": seed,
                            "split": split,
                            "train_seconds": seconds,
                            **scores(data[split]["label"], predictions[split]),
                        }
                    )
                prediction_rows.extend(
                    {
                        "model": model,
                        "train_per_class": size,
                        "seed": seed,
                        "text": text,
                        "true_label": LABELS[truth],
                        "predicted_label": LABELS[prediction],
                        "correct": truth == prediction,
                    }
                    for text, truth, prediction in zip(
                        data["test"]["text"], data["test"]["label"], predictions["test"]
                    )
                )
                pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
                pd.DataFrame(prediction_rows).to_csv(
                    args.output_dir / "predictions.csv", index=False
                )

    plot_learning_curve(pd.DataFrame(rows), args.output_dir / "learning_curve.png")


if __name__ == "__main__":
    main()
