# Nonsymmetric DPP Learning
PyTorch implementation of nonsymmetric determinantal point process (DPP) learning.  For 
details, see our [Learning Nonsymmetric Determinantal Point Processes](https://arxiv.org/abs/1905.12962)
paper ([NeurIPS 2019 poster](https://drive.google.com/file/d/1zM3dAqcskQFndZN8yTiMFS6s2rnjzVYB/view?usp=sharing)).

## Installation
Install required Python packages:
```console
$ pip install --user --requirement requirements.txt
```

## Usage
Train and evaluate a nonsymmetric DPP model using the Amazon apparel baby registry 
dataset:
```console
$ cd src
$ python main.py --dataset_name basket_ids --input_file data/1_100_100_100_apparel_regs.csv --num_sym_embedding_dims 30 --num_nonsym_embedding_dims 30
```

For a full list of command-line options:
```console
$ python main.py --help
```
