"""
Synopsis: Some useful functions.
"""
import sys
import os

Header = os.path.dirname(os.path.abspath(__file__))
Header = Header[:-3]
sys.path.append(Header)

import random
import tempfile
import glob
import io
import codecs
import logging
import argparse
import sqlite3

import numpy as np
import pandas as pd

import torch

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# control random-number generators
torch.manual_seed(1234)
random.seed(1234)

# Set default for floating point to torch.float64
torch.set_default_tensor_type(torch.DoubleTensor)

# Offset added to det(L_i) term in nonsymmetric low-rank DPP log-likelihood, to promote
# positive-definiteness and improve numerical stability for Cholesky decomposition
epsilon = 1e-5

class LogLikelihood(object):

    @staticmethod
    def compute_log_likelihood(model, baskets, alpha_regularization=0.,
                               beta_regularization=0.,
                               gamma_regularization=0.,
                               reduce=True, checks=False, mapped=True):
        """
        Computes nonsymmetric low-rank DPP log-likelihood
        """
        num_baskets = len(baskets)
        batchnorm = "BatchNorm" in str(model.embeddings)

        # Get the symmetric and nonsymmetric embedding components of each product in the catalog
        B = None
        C = None
        if model.disable_nonsym_embeddings:
            V = model.forward(model.all_items_in_catalog_set_var)
        else:
            V, B, C = model.forward(model.all_items_in_catalog_set_var)

        # get embeddings for each basket
        V_embeddings = [V[basket] for basket in baskets]

        if not model.disable_nonsym_embeddings:
            B_embeddings = [B[basket] for basket in baskets]
            C_embeddings = [C[basket] for basket in baskets]

        # Compute first term (numerator) of nonsymmetric low-rank DPP likelihood
        if reduce:
            first_term = 0
        else:
            first_term = torch.zeros(num_baskets).to(model.device)

        for i, V_i in enumerate(V_embeddings):
            # Symmetric component
            L_i_symm = V_i.mm(V_i.transpose(0, 1))

            # Nonsymmetric components
            if not model.disable_nonsym_embeddings:
                B_i = B_embeddings[i]
                C_i = C_embeddings[i]
                nonsymm_i = B_i.mm(C_i.transpose(0, 1)) - C_i.mm(B_i.transpose(0, 1))

            # Add epsilon * I to improve numerical stability
            eye_L_i = torch.eye(L_i_symm.size()[0]).to(model.device)
            if model.disable_nonsym_embeddings:
                tmp = torch.slogdet(L_i_symm + epsilon * eye_L_i)[1]
            else:
                tmp = torch.slogdet(L_i_symm + epsilon * eye_L_i + nonsymm_i)[1]

            tmp = tmp.to(model.device)
            if reduce:
                first_term += tmp
            else:
                first_term[i] = tmp

        # Compute denominator of nonsymmetric low-rank DPP likelihood (normalization constant)
        # Symmetric component
        if model.disable_nonsym_embeddings:
            # Use dual form of L when nonsymmetric component is disabled
            L_dual = V.transpose(0, 1).mm(V)
            L = L_dual

            num_sym_embedding_dims = L_dual.size(0)
            identity = torch.eye(num_sym_embedding_dims).to(model.device)
        else:
            L = V.mm(V.transpose(0, 1))
            num_catalog_items = L.size(0)
            identity = torch.eye(num_catalog_items).to(model.device)

        if not model.disable_nonsym_embeddings:
            # Nonsymmetric component
            nonsymm = B.mm(C.transpose(0, 1)) - C.mm(B.transpose(0, 1))

        # don't forget smooth the normalization term too (lest DPP is no longer
        # a probability density)
        if batchnorm:
            second_term = 0
        else:
            if model.disable_nonsym_embeddings:
                logpartition = torch.slogdet(L + identity)[1]
            else:
                logpartition = torch.slogdet(L + nonsymm + identity)[1]
            second_term = logpartition.to(model.device)

        # L2-style regularization
        third_term = None
        if alpha_regularization != 0 or \
                beta_regularization != 0 or \
                gamma_regularization != 0:
            third_term = model.reg(
                V, B, C, model.lambda_vec,
                torch.Tensor([alpha_regularization]),
                torch.Tensor([beta_regularization]),
                torch.Tensor([gamma_regularization]))
        else:
            third_term = 0.

        # if reduce is set, then at this point logliks holds the sum of logliks
        # over all baskets in this minibatch, else it's just a list of the
        # latter
        if reduce:
            first_term /= num_baskets  # this now the avg loglik over all bsks
            logliks = first_term - second_term - third_term
        else:
            logliks = first_term - second_term - third_term

        if checks:
            if reduce and alpha_regularization == 0.:
                assert logliks <= 0

        return logliks


class VocabularyMapper(object):
    """
    Maps categorical values onto indices in a vocabulary
    """
    def __init__(self, vocab):
        self.vocab = np.unique(vocab)
        self.vocab.sort()

    def __call__(self, values):
        return np.searchsorted(self.vocab, values)


class PackedLoggers(object):
    """
    Combine a bunch of loggers into 1.
    """
    def __init__(self, loggers):
        self.loggers = loggers

    def add_scalar(self, *args, **kwargs):
        for logger in self.loggers:
            logger.add_scalar(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        for logger in self.loggers:
            if hasattr(logger, "add_histogram"):
                logger.add_histogram(*args, **kwargs)

    def add_embedding(self, model, val_data, **kwargs):
        out = model.forward(val_data)
        out = torch.cat((out.data, torch.ones(len(out), 1)), 1)

        for logger in self.loggers:
            if hasattr(logger, "add_embedding"):
                self.logger.add_embedding(
                    out, metadata=out.data, label_img=val_data.data.double(),
                    **kwargs)

    def new_iteration(self):
        for logger in self.loggers:
            if hasattr(logger, "new_iteration"):
                logger.new_iteration()

    def model_checkpoint(self, model, **kwargs):
        for logger in self.loggers:
            if hasattr(logger, "model_checkpoint"):
                logger.model_checkpoint(model, **kwargs)


def str2bool(v):
    """
    Converts a user-supplied yes/no response to boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(s, separator=",", transform=float):
       """
       Convert comma-separated string into list
       """
       if not s:
           return []
       return list(map(transform, s.split(separator)))


def str2loi(s, separator=","):
    return str2list(s, separator=separator, transform=int)



def parse_cmdline_args():
    """
    Parses command-line arguments / options for this software.
    """
    parser = argparse.ArgumentParser(
        description='Train a symmetric or nonsymmetric DPP',
        epilog=("Example usage: python main.py --dataset_name basket_ids" 
                "--input_file data/1_100_100_100_apparel_regs.csv"  
                "--num_sym_embedding_dims 30 --num_nonsym_embedding_dims 30"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--hogwild', type=str2bool, default="false",
        help='whether to enable HogWild parallel training')
    parser.add_argument("--inference", type=str2bool,
                        default=True, help="run inference on val / test data")
    parser.add_argument("--tsne", action="store_true", default=False,
                        help="do t-SNE projections of embeddings")
    parser.add_argument("--scores_file", type=str,
                        default="nonsymmetric-DPP-eval-scores",
                        help="pickle file where inference scores will be written (pandas dataframe format)")
    parser.add_argument(
        '--num_bootstraps', type=int, default=1,
        help='number of bootstraps for evaluation scores')
    parser.add_argument("--disable_eval", type=str2bool, default="true",
                        help="disable model evaluation during training")
    parser.add_argument(
        '--batch_size', type=int, default=200,
        help='batch size for creating training data')
    parser.add_argument(
        '--input_file', type=str, default=None,
        help='input file path')
    parser.add_argument(
        '--input_file_test_negatives', type=str, default=None,
        help='input file test negatives')
    parser.add_argument(
        '--disjoint_sets_file_w', type=str, default=None,
        help='input file  disjoint_sets_file_w')
    parser.add_argument(
        '--input_file_disjoint_sets', type=str, default=None,
        help='input file  input_file_disjoint_sets')

    parser.add_argument(
        '--num_iterations', type=int, default=1000,
        help='number of passes to do over data during training')
    parser.add_argument(
        '--num_baskets', type=int,
        help='number of baskets to use in experiment (limits catalog size)')
    parser.add_argument(
        '--max_basket_size', type=int, default=np.inf,
        help='maximum size of the baskets to use in experiment')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='L2 regularization parameter for symmetric component')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='L2 regularization parameter for nonsymmetric component')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='L2 regularization parameter for nonsymmetric component')
    parser.add_argument(
        '--use_metadata', type=str2bool, default="false",
        help='whether to use product meta-data to enrich embeddings')
    parser.add_argument(
        '--use_price', type=str2bool, default="false",
        help='whether to use product price meta-data to enrich embeddings')
    parser.add_argument(
        '--use_fasttext', type=str2bool, default="false",
        help='whether to use product description FastText to enrich embeddings')
    parser.add_argument(
        '--prepend_meta', type=str2bool, default="true",
        help='whether to include meta-data before or after computing embedding')
    parser.add_argument(
        '--num_threads', type=int, default=1,
        help='num_threads to use for intra-process parallelism')
    parser.add_argument(
        '--db_path', required=False, default="logs.db",
        help='path to db where `pyml_experiments` logs will be written')
    parser.add_argument(
        '--disable_gpu', type=str2bool, default="false",
        help='disable gpu usage')

    dataset_parser = parser.add_argument_group("dataset specification options")
    dataset_parser.add_argument(
        '--dataset_name', type=str,
        default="basket_ids", help='Name of the dataset to use.  Currently either "basket_ids" or "uk" is supported.')

    model_parser = parser.add_argument_group("model / optimizer options")
    model_parser.add_argument('--hidden_dims', type=str2loi, default="",
                              help=('comma separated list of hidden layer '
                                    'dimensions'))
    model_parser.add_argument(
        '--num_sym_embedding_dims', type=int, default=30,
        help='number of final embedding dims for symmetric kernel component to use')
    model_parser.add_argument(
        '--num_nonsym_embedding_dims', type=int, default=100,
        help='number of final embedding dims for nonsymmetric kernel component to use')
    model_parser.add_argument(
        '--product_id_embedding_dim', type=int, default=30,
        help='number of product id embeddings dims to use')
    model_parser.add_argument(
        '--aisle_id_embedding_dim', type=int, default=20,
        help='number of aisle id embeddings dims to use(currently used for Instacart dataset only)')
    model_parser.add_argument(
        '--department_id_embedding_dim', type=int, default=20,
        help='number of department id embeddings dims to use(currently used for Instacart dataset only)')
    model_parser.add_argument(
        '--learning_rate', type=float, default=0.1,
        help='initial learning rate for optimizer')
    # model_parser.add_argument(
    #     '--optimizer', choices=["adam", "adagrad", "sgd", "rmsprop"], type=str,
    #     default="adam", help='optimizer to use training the model')
    model_parser.add_argument(
        '--activation', choices=["selu", "relu", "tanh"], type=str,
        default="selu", help='non-linear activation to use')
    model_parser.add_argument(
        '--dropout', type=float, default=0,
        help='amount of dropout to use')
    model_parser.add_argument(
        '--persisted_model_dir', type=str, default="saved_models",
        help='Path to the dir where model will be/was persisted. ')
    model_parser.add_argument(
        '--num_val_baskets', type=int, default=300)
    model_parser.add_argument(
        '--num_test_baskets', type=int, default=2000)

    args = parser.parse_args()

    # sanitize some arguments which have ranges
    if args.hogwild and args.num_threads < 2:
        raise ValueError("--hogwild true but --num_threads 1 < 2")
    if args.inference and args.scores_file is None:
        raise ValueError("no --scores_file specified with --inference")

    args.product_id_embedding_dim = args.num_sym_embedding_dims

    args.scores_file = Header + args.scores_file
    args.persisted_model_dir = Header + args.persisted_model_dir

    if args.input_file is None and args.dataset_name == "basket_ids":
        args.input_file = "data/belgian-retail.csv"
    if args.input_file is not None:
        args.input_file = Header + args.input_file

    if args.input_file_test_negatives is not None:
        args.input_file_test_negatives = Header + args.input_file_test_negatives

    if args.disjoint_sets_file_w is not None:
        args.disjoint_sets_file_w = Header + args.disjoint_sets_file_w

    if args.input_file_disjoint_sets is not None:
        args.input_file_disjoint_sets = Header + args.input_file_disjoint_sets

    return args


