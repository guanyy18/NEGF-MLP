# NEGF-MLP

Machine-learning interatomic potentials for voltage-biased Li-water system, trained on NEGF-DFT data.

## Structure

```
NEGF-MLP/
├── module/            # shared modules
│   ├── interaction.py # equivariant MACE interaction blocks
│   └── get_dataset.py # trajectory NPZ -> torch_geometric Dataset
├── base/              # 0 V base model
│   ├── model.py
│   ├── node.py
│   └── train.py
├── delta/             # voltage-dependent delta model
│   ├── model_delta.py
│   ├── node_delta.py
│   └── train_delta.py
└── dataset/           # training data (not tracked by git)
```

## Setup

Requires PyTorch, torch_geometric, e3nn, ASE.

## Data

The training data (`dataset/*.npz`, NPZ of NEGF-DFT trajectories) is not included in this repository due to size. Contact the author for access, or place your own NEGF-DFT trajectory NPZ files in `dataset/` before training.

## Usage

```bash
# Base model (0 V)
python base/train.py

# Delta model (voltage-dependent)
python delta/train_delta.py
```
