# Python4NLP final project

An experiment comparing TF–IDF logistic regression with fine-tuned
DistilBERT on low-data, four-class emotion classification. It uses the
single-label `joy`, `anger`, `sadness`, and `fear` examples from GoEmotions.
GoEmotions is distributed under the Apache 2.0 licence; see its
[dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions)
and [paper](https://aclanthology.org/2020.acl-main.372/).

## Setup
Run the experiment:

```bash
uv run python -m src.run
```

Results are written to `results/metrics.csv`, `results/predictions.csv`, and
`results/learning_curve.png`. The predictions file supports manual error
analysis. The first run downloads GoEmotions and DistilBERT from Hugging Face.