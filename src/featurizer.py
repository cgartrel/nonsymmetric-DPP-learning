"""

"""
import typing
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from mlp import MLP


class ProductCatalogEmbedder(nn.Module):
    """
    Featurizer for product catalogs.

    Parameters
    ----------
    features_setup:
    """
    def __init__(self, catalog: pd.DataFrame,
                 features_setup: typing.Dict[int, typing.Dict],
                 output_dim: int, hidden_dims: typing.List[int]=None,
                 activation: str="relu", dropout: float=0,
                 batchnorm: bool=False, device=None):
        super(ProductCatalogEmbedder, self).__init__()
        self.catalog = catalog
        self.features_setup = features_setup
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.device = device
        self._compile()

    def _compile(self):
        for col_name in self.catalog:
            if "fasttext" in col_name:
                self.features_setup[col_name] = {}

        self.catalog_ = {}
        input_dim = 0
        for feature_name, setup in self.features_setup.items():
            feature = self.catalog[feature_name].tolist()
            embedding_dim = setup.get("embedding_dim", None)
            if embedding_dim is None:
                feature = torch.DoubleTensor(feature)
                embedder = None
                input_dim += 1
            elif embedding_dim == "log":
                feature = torch.DoubleTensor(feature)
                feature = torch.log(feature)
                embedder = None
                input_dim += 1
            else:
                feature = torch.LongTensor(feature)
                if feature_name == "product_id":
                    print(feature.min())
                embedder = nn.Embedding(feature.max() + 1,
                                        embedding_dim)
                if self.device is not None:
                    embedder.to(self.device)
                embedder.weight.data.uniform_(0.0, 1.0)
                input_dim += embedding_dim
            if self.device is not None and isinstance(feature, torch.Tensor):
                feature = feature.to(self.device)
            self.catalog_[feature_name] = feature
            setattr(self, "%s_embedding_layer" % feature_name, embedder)

            if self.hidden_dims is None:
                self.activation = None
                self.hidden_dims = []

        self.mlp = MLP(input_dim, self.output_dim,
                       activation=self.activation,
                       hidden_dims=self.hidden_dims,
                       dropout=self.dropout,
                       batchnorm=self.batchnorm)

    def forward(self, return_all=False):
        embeddings = []
        all_embeddings = {}
        for feature_name in self.features_setup:
            feature = self.catalog_[feature_name]
            embedding_layer = getattr(self, "%s_embedding_layer" % (
                feature_name))
            if embedding_layer is None:
                embedding = feature.unsqueeze(1)
            else:
                embedding = embedding_layer(feature)
            embeddings.append(embedding)
            if return_all:
                all_embeddings[feature_name] = embedding
        embedding = torch.cat(embeddings, dim=-1)

        if self.mlp is not None:
            embedding = self.mlp(embedding)

        if return_all:
            return embedding, all_embeddings
        else:
            return embedding
