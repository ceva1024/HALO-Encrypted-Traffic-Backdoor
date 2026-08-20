# Demo data

The preprocessed demo splits used by this repository are distributed
separately and are not tracked in Git.

## Download

Download [HALO demo data v1.0](https://drive.google.com/file/d/1daYv-3oepF3YIQvZfnb6GaVyTUa9zhC-/view?usp=sharing)
from Google Drive.

## Setup

Extract the three JSONL files into this directory. From the repository root,
the expected layout is:

```text
data/
├── README.md
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

The commands in [experiments/RUN.md](../experiments/RUN.md) use these paths
directly.

## Provenance and terms

These demo splits are derived from the CSTNET-TLS 1.3 dataset introduced with
ET-BERT and have been preprocessed for the packet-sequence representation used
by this repository. See [REFERENCES.md](../REFERENCES.md) for the full data
reference.