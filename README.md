# Python4NLP final project

An experiment comparing TF–IDF logistic regression with fine-tuned
DistilBERT on low-data, four-class emotion classification. It uses the
single-label `joy`, `anger`, `sadness`, and `fear` examples from GoEmotions.
GoEmotions is distributed under the Apache 2.0 licence. See its
[dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions)
and [paper](https://aclanthology.org/2020.acl-main.372/).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- a system supported by PyTorch
- internet access on the first run to download GoEmotions and DistilBERT from
  Hugging Face

The project requires Python 3.13 or newer. Package versions are recorded in
`uv.lock`.

## Reproducing the experiment

From the repository root, run:

```bash
uv run python -m src.run
```

By default, both models use 5, 10, 15, 20, 25, 50, 100, and 200 training
examples per class across three seeds. DistilBERT is fine-tuned for 10 epochs.
For each seed, larger training sets contain all examples from the smaller sets.

Results are written to `results/metrics.csv`, `results/predictions.csv`, and
`results/learning_curve.png`. The predictions file supports manual error
analysis. Existing files with these names are overwritten.
The committed result files contain the outputs used in the report.

## Report

The report source is in `report/`. With a LaTeX installation that includes
`latexmk`, build the PDF with:

```bash
make -C report
```
