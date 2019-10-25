# Nonsymmetric DPP Learning
Pytorch implementation of nonsymmetric determinantal point process (DPP) learning.  For 
details, see our [Learning Nonsymmetric Determinantal Point Processes](https://arxiv.org/abs/1905.12962)
paper.

## Installation
Create and activate a new Python virtual environment (recommended):
```console
$ virtualenv --system-site-packages -p python3 ./venv
$ source ./venv/bin/activate
```

Install required Python packages:
```console
(venv)$ pip install --user --requirement requirements.txt
```

## Usage
Train and evaluate a nonsymmetric DPP model using the Amazon apparel baby registry 
dataset:
```console
(venv)$ cd src
(venv)$ python main.py --dataset_name basket_ids --input_file data/1_100_100_100_apparel_regs.csv --num_sym_embedding_dims 30 --num_nonsym_embedding_dims 30
```

For a full list of command-line options:
```console
(venv)$ python main.py --help
```
