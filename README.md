# HALO-Encrypted-Traffic-Backdoor

## Paper

This repository is the official implementation of:

Shaotong Wang, Jiaxuan Geng, Yingjie Zhou, Song Yang, Xing Yang, and
Dapeng Oliver Wu, “HALO: Revealing the inadequacy of backdoor defenses
in encrypted traffic classification,” Computer Networks, vol. 288,
article 112656, 2026.

https://doi.org/10.1016/j.comnet.2026.112656

## Repository layout

```text
data/                            small preprocessed demo splits
halo/                            model, attack, and defense implementation
detect/                          defense entry points
experiments/                     experiment workflow
train/                           training command-line entry points
```

## Installation

The release was tested with Python 3.9.19, NumPy 1.24.3, PyTorch 2.0.1,
Matplotlib 3.7.1, and scikit-learn 1.3.0. A CUDA-enabled PyTorch installation is
recommended for training; all entry points also support `--cpu`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

The repository includes a small preprocessed dataset for testing the training
and evaluation workflow. The commands are provided in the
[RUN.md](experiments/RUN.md). Run them in this order:

1. train the clean classifier;
2. train HALO;
3. train BadNets, TrojanFlow, and UAP; and
4. run SCAn, Beatrix, and TED against every attack.

The commands use relative paths and write checkpoints to `models/` and logs to
`logs/`. Set `CUDA_VISIBLE_DEVICES` to choose a GPU.

## Full workflow

To run the complete workflow, start from the original packet captures and
prepare the data before training:

```text
PCAP files
    ↓
flow extraction and labeling
    ↓
packet sequence construction
    ↓
train/validation/test split
    ↓
training and evaluation
```

## Responsible use

This code is intended for authorized robustness research on encrypted-traffic
classifiers. Do not use it to interfere with networks, services, or data that
you do not own or have explicit permission to test. See [SECURITY.md](SECURITY.md).

## License

Except for third-party materials and the contents of `data/`, the original
source code and documentation in this repository are licensed under the
Apache License, Version 2.0. See [LICENSE](LICENSE).
