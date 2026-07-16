import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from .config import LABELS


def load_data() -> DatasetDict:
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = raw["train"].features["labels"].feature.names
    source_ids = {names.index(label): i for i, label in enumerate(LABELS)}

    def relevant(example):
        return len(example["labels"]) == 1 and example["labels"][0] in source_ids

    def convert(example):
        return {"text": example["text"], "label": source_ids[example["labels"][0]]}

    return DatasetDict(
        {
            split: dataset.filter(relevant).map(
                convert, remove_columns=dataset.column_names
            )
            for split, dataset in raw.items()
        }
    )


def sample_per_class(dataset: Dataset, size: int, seed: int) -> Dataset:
    labels = np.asarray(dataset["label"])
    rng = np.random.default_rng(seed)
    class_indices = [
        rng.permutation(np.flatnonzero(labels == label))
        for label in range(len(LABELS))
    ]
    if any(size > len(indices) for indices in class_indices):
        raise ValueError(f"Cannot sample {size} examples from every class")
    indices = np.concatenate([indices[:size] for indices in class_indices])
    rng.shuffle(indices)
    return dataset.select(indices.tolist())
