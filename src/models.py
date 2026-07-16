from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as transformers_logging

from .config import LABELS, MAX_LENGTH

transformers_logging.set_verbosity_error()


def scores(labels, predictions) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    } # type: ignore


def train_tfidf(train, validation, test):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1_000),
    )
    start = perf_counter()
    model.fit(train["text"], train["label"])
    elapsed = perf_counter() - start
    predictions = {
        split: model.predict(data["text"])
        for split, data in {"validation": validation, "test": test}.items()
    }
    return elapsed, predictions


def train_transformer(
    train, validation, test, model_name: str, epochs: int, seed: int
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    encoded = {
        split: data.map(tokenize, batched=True, remove_columns=["text"])
        for split, data in {
            "train": train,
            "validation": validation,
            "test": test,
        }.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    with TemporaryDirectory() as output_dir:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                learning_rate=2e-5,
                weight_decay=0.01,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=64,
                save_strategy="no",
                logging_strategy="no",
                disable_tqdm=False,
                dataloader_pin_memory=False,
                report_to="none",
                seed=seed,
                data_seed=seed,
            ),
            train_dataset=encoded["train"],
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        start = perf_counter()
        trainer.train()
        elapsed = perf_counter() - start
        predictions = {
            split: trainer.predict(encoded[split]).predictions.argmax(axis=1)  # type: ignore
            for split in ("validation", "test")
        }
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return elapsed, predictions
