# Scalable nonsymmetric DPP Learning
PyTorch implementation of scalable nonsymmetric determinantal point process (NDPP) learning.  For 
details, see our [Scalable Learning and MAP Inference for Nonsymmetric Determinantal Point Processes](https://arxiv.org/abs/2006.09862)
paper.

## Installation
Install required Python packages:
```console
$ pip install --user --requirement requirements.txt
```

## Usage
Train and evaluate a scalable NDPP model using the Amazon apparel baby registry 
dataset:
```console
$ cd src
$ python main.py --dataset_name basket_ids --input_file data/1_100_100_100_apparel_regs.csv --num_sym_embedding_dims 30 --num_nonsym_embedding_dims 30 --alpha 0 --batch_size 800
```

Train and evaluate a scalable NDPP using the UK retail dataset:
```console
$ python main.py --dataset_name uk --num_sym_embedding_dims 100 --num_nonsym_embedding_dims 100 --alpha 0.001 --max_basket_size 100
```

To train and evaluate a scalable NDPP using the Instacart dataset, first download the Instacart dataset 
from https://www.instacart.com/datasets/grocery-shopping-2017 and unpack it in the 
`data/instacart_2017_05_01` directory.  Next, unzip the `data/instacart_2017_05_01/meta_catalog.csv.zip` file.  Then run:
```console
$ python main.py --dataset_name instacart --num_sym_embedding_dims 100 --num_nonsym_embedding_dims 100 --alpha 0.01 --max_basket_size 100 --batch_size 800
```

To train and evaluate a scalable NDPP using the Million Song dataset, first download the dataset 
from http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip and unpack `train_triples.txt` file in the `data` directory. Then run:
```console
$ python main.py --dataset_name millionsong --num_sym_embedding_dims 150 --num_nonsym_embedding_dims 150 --alpha 0.01 --max_basket_size 150 --batch_size 400
```

For a full list of command-line options:
```console
$ python main.py --help
```
