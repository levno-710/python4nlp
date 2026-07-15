from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(results: pd.DataFrame, path: Path) -> None:
    test = results[results["split"] == "test"]
    summary = test.groupby(["model", "train_per_class"])["macro_f1"].agg(
        ["mean", "std"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, values in summary.groupby(level="model"):
        values = values.droplevel("model")
        ax.errorbar(
            values.index,
            values["mean"],
            yerr=values["std"].fillna(0),
            marker="o",
            capsize=3,
            label=model,
        )
    ax.set(
        xlabel="Training examples per class", ylabel="Test macro-F1", ylim=(0, 1)
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
