"""
Synopsis: Refactored nonsymmetric DPP code logic

"""
import os
import random

import numpy as np
import pandas as pd
from results import Results

from sklearn.utils import check_random_state
from sklearn.manifold import TSNE

import torch

from utils import (logging, parse_cmdline_args)
from featurizer import ProductCatalogEmbedder
from datasets import (load_dataset, BasketDataLoader)
from nonsymmetric_dpp_learning import (L2Regularization,
                                        do_learning, eval_model, compute_log_likelihood)
from nonsymmetric_dpp_prediction import NonSymmetricDPPPrediction

# control random-number generators
torch.manual_seed(1234)
random.seed(1446)
np.random.seed(13564)

# Set default for floating point to torch.float64
torch.set_default_tensor_type(torch.DoubleTensor)


class NonSymmetricDPP(NonSymmetricDPPPrediction):
    """
    (Deep) Nonsymmetric Low Rank Determinantal Point Processes.
    """
    def __init__(self, product_catalog,
                 num_sym_embedding_dims=None, num_nonsym_embedding_dims=None,
                 features_setup={"product_id": {"num_sym_embedding_dims": 100,
                                                "num_nonsym_embedding_dims": 10}},
                 disable_gpu=False, epsilon=1e-5,
                 hidden_dims=None, activation="selu", logger=None,
                 random_state=None, dropout=None, **kwargs):
        super(NonSymmetricDPP, self).__init__(**kwargs)
        self.product_catalog = product_catalog
        self.num_sym_embedding_dims = num_sym_embedding_dims
        self.num_nonsym_embedding_dims = num_nonsym_embedding_dims
        self.features_setup = features_setup
        self.disable_gpu = disable_gpu
        self.epsilon = epsilon
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.random_state = check_random_state(random_state)
        self.dropout = dropout
        self._compile()

    def _compile(self):
        # build product embedder
        self.get_v_embeddings = ProductCatalogEmbedder(
            self.product_catalog, self.features_setup, self.num_sym_embedding_dims,
            activation=self.activation, hidden_dims=self.hidden_dims,
            dropout=self.dropout)

        if (self.num_nonsym_embedding_dims == 0):
            logging.info("num_nonsym_embedding_dims = 0; disabling non-symmetric components")
            self.disable_nonsym_embeddings = True
        else:
            self.get_b_embeddings = self.get_v_embeddings
            self.d_params = torch.randn(
                self.num_nonsym_embedding_dims,
                self.num_nonsym_embedding_dims, requires_grad=True)

        # prepare L2 regularizer
        self.reg = L2Regularization().regularization

        # determine whether to use GPU or CPU
        if not self.disable_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        logging.info("Using device: %s " % self.device)

        # XXX for backward compat
        self.all_items_in_catalog = self.product_catalog.product_id.unique().tolist()
        self.all_items_in_catalog_set = set(self.all_items_in_catalog)
        self.item_catalog_size = len(self.all_items_in_catalog_set)
        self.all_items_in_catalog_set_var = torch.LongTensor(
            np.arange(self.item_catalog_size)).to(self.device)

        # compute item counts
        self.lambda_vec = []
        self.item_counts = self.product_catalog.product_id.value_counts().to_dict()
        for item_id, count in self.item_counts.items():
            if item_id in self.item_counts:
                self.lambda_vec.append(1. / count)
            else:
                self.lambda_vec.append(1.)
        self.lambda_vec = torch.Tensor(self.lambda_vec).to(self.device)

    def embeddings(self):
        """
        XXX For backward compat
        """
        if self.disable_nonsym_embeddings:
            return self.get_v_embeddings().to(self.device)
        else:
            return self.get_v_embeddings().to(self.device), \
                   self.get_b_embeddings().to(self.device)

    def forward(self, _):
        """
        XXX For backward compat
        """
        if self.disable_nonsym_embeddings:
            return self.get_v_embeddings().to(self.device)
        else:
            return self.get_v_embeddings().to(self.device), \
                   self.get_b_embeddings().to(self.device), \
                   self.d_params.to(self.device)

    @staticmethod
    def compute_log_likelihood(model, baskets, alpha_regularization=0.,
                               beta_regularization=0.,
                               reduce=True, checks=False, mapped=True):
        return compute_log_likelihood(model, baskets, alpha_regularization=alpha_regularization,
                           beta_regularization=beta_regularization,
                           reduce=reduce, checks=checks, mapped=mapped)

    def get_tsne_embeddings(self, n_components=2, **kwargs):
        tsne = TSNE(n_components=n_components, random_state=self.random_state,
                    **kwargs)
        embeddings = self.get_v_embeddings().data.numpy()
        tsne_embeddings = tsne.fit_transform(embeddings)
        return tsne_embeddings



def prepare_data(args, random_state=None, num_val_baskets=None,
                 num_test_baskets=None,
                 max_basket_size=np.inf):
    rng = check_random_state(random_state)
    ds = load_dataset(dataset_name=args.dataset_name,
                      num_baskets=args.num_baskets,
                      use_metadata=args.use_metadata,
                      random_state=rng,
                      max_basket_size=max_basket_size,
                      input_file=args.input_file)

    # train / val / test split
    logging.info("Spliting dataset into train / val / test")
    num_train_baskets = len(ds.baskets) - num_val_baskets - num_test_baskets
    train_ds, val_ds, test_ds = ds.split([num_train_baskets, num_val_baskets,
                                          num_test_baskets])
    logging.info("%i train baskets" % len(train_ds.baskets))
    basket_sizes = [len(x) for x in train_ds.baskets]
    logging.info("number items in training baskets, avg:%d, variance: %d" %
                 (np.mean(basket_sizes), np.var(basket_sizes)))
    logging.info("%i val baskets" % len(val_ds.baskets))

    # build mini-batch generator
    logging.info("%i test baskets" % len(test_ds.baskets))
    train_data_loader = BasketDataLoader(train_ds, batch_size=args.batch_size)

    return (ds.product_catalog, ds.get_basket_size_buckets(),
            train_data_loader, val_ds, test_ds)


class Args(object):

    @staticmethod
    def get_default_cli_args():
        return parse_cmdline_args()

    @staticmethod
    def build_from_cli():
        return Args(parse_cmdline_args())

    def __init__(self, args):
        # command-line arguments
        self.args = args
        self.args_dict = vars(self.args)
        self.hidden_dims = self._compute_hidden_dims(self.args)
        self.lr = self._infer_learning_rate(self.args, self.hidden_dims)
        self.alpha = self._compute_alpha(self.args, self.hidden_dims)
        # self.beta = self._compute_beta(self.args, self.hidden_dims)
        self.beta = 0 # Beta regularization hyperparam is currently not used
        self.disable_eval = self.args_dict.pop("disable_eval")
        self.inference = self.args_dict.pop("inference")
        self.num_bootstraps = self.args_dict.pop("num_bootstraps")
        self.tsne = self.args_dict.pop("tsne")
        for param, value in self.args_dict.items():
            if value is not None:
                logging.info(".....args.%s: %s" % (param, value))

    def compute_features_setup(self, product_catalog):
        args = self.args
        # define feature "transforms"
        features_setup = {
            "product_id": {"embedding_dim": args.product_id_embedding_dim}
        }

        if args.use_metadata:
            if "aisle_id" in product_catalog.columns:
                features_setup["aisle_id"] = {"embedding_dim": args.aisle_id_embedding_dim}
            if "department_id" in product_catalog.columns:
                features_setup["department_id"] = {"embedding_dim": args.department_id_embedding_dim}
        return features_setup

    @staticmethod
    def _compute_hidden_dims(args):
        hidden_dims = args.hidden_dims
        if hidden_dims is None:
            hidden_dims = []
        return hidden_dims

    @staticmethod
    def _infer_learning_rate(args, hidden_dims):
        logging.info("Hyper-parameters:")

        if args.learning_rate is None:
            if len(hidden_dims) < 2:
                lr = 0.1
            else:
                lr = 0.01
            logging.info(".....learning_rate: %g" % lr)
        else:
            lr = args.learning_rate
        return lr

    @staticmethod
    def _compute_alpha(args, hidden_dims):
        # it's important that the shallow model penalized embeddings
        alpha = args.alpha
        if alpha is None:
            if len(hidden_dims) == 0:
                alpha = 1.
            else:
                alpha = 0.
            logging.info(".....alpha: %g" % alpha)
        return alpha

    @staticmethod
    def _compute_beta(args, hidden_dims):
        # it's important that the shallow model penalized embeddings
        beta = args.beta
        if beta is None:
            if len(hidden_dims) == 0:
                beta = 1.
            else:
                beta = 0.
            logging.info(".....beta: %g" % beta)
        return beta

class Dataset(object):
    def __init__(self, args, seed, rng, num_val_baskets, num_test_baskets):
        # grab dataset
        (product_catalog, basket_size_buckets, train_data,
         val_data, test_data) = prepare_data(args, random_state=rng,
                                   num_val_baskets=num_val_baskets,
                                   num_test_baskets=num_test_baskets,
                                   max_basket_size=args.max_basket_size)
        self.seed = seed
        self.num_val_baskets = num_val_baskets
        self.num_test_baskets = num_test_baskets
        self.product_catalog = product_catalog
        self.basket_size_buckets = basket_size_buckets
        self.max_basket_size = max(self.basket_size_buckets.keys())
        self.train_data = train_data
        self.val_data = val_data.baskets
        self.test_data = test_data.baskets

class Experiment(object):

    @classmethod
    def build(cls, arguments, dataset):
        args = arguments.args
        logging.info("Building model for %s" % (args.scores_file,))
        model = cls._build_model_object(arguments, dataset.product_catalog,
                                        dataset.max_basket_size,
                                        dataset.seed)
        ofile = cls._load_model(arguments, model, dataset)
        return model, ofile

    @staticmethod
    def save_tsne_projections(model, ofile):
        logging.info("Doing t-SNE projections")
        tsne_embeddings = model.get_tsne_embeddings()
        tsne_ofile = "%s.tsne" % ofile
        np.savetxt(tsne_ofile, tsne_embeddings)
        logging.info("Save t-SNE projections to file %s" % tsne_ofile)

    @staticmethod
    def run(model, arguments, dataset, store_inference_scores=False):
        args = arguments.args
        args_dict = arguments.args_dict

        logging.info("Running inference on test data for %s" % (args.scores_file,))
        artifacts, _ = eval_model(model, dataset.val_data, inference=arguments.inference,
                                  test_data=dataset.test_data, end=True,
                                  buckets=dataset.basket_size_buckets,
                                  num_threads=args.num_threads,
                                  num_bootstraps=arguments.num_bootstraps)

        scores = artifacts["scores"]
        df = pd.DataFrame(scores)
        for param, value in args_dict.items():
            if param == "hidden_dims":
                value = ",".join(list(map(str, arguments.hidden_dims)))
            df[param] = value
        logging.info("Scores:")
        print(df)
        pid = os.getpid()
        if store_inference_scores:
            # store scores unto disk
            scores_file = "%s.%i" % (args.scores_file, pid)
            df.to_pickle(scores_file)
            logging.info("Inference scores written to %s" % scores_file)
        logging.info("Process %i complete." % pid)
        return df

    @classmethod
    def _build_model_object(cls, arguments, product_catalog, max_basket_size,
                            seed):
        args = arguments.args
        model_cls = NonSymmetricDPP
        model_params = {param: getattr(args, param)
                        for param in ["hidden_dims",
                                      "activation",
                                      "disable_gpu",
                                      "dropout"
                        ]}
        model_params["num_sym_embedding_dims"] = cls._compute_num_sym_embeddings(args)
        model_params["num_nonsym_embedding_dims"] = cls._compute_num_nonsym_embeddings(args)
        features_setup = arguments.compute_features_setup(product_catalog)
        model = model_cls(product_catalog, features_setup=features_setup,
                          **model_params)

        if args.num_nonsym_embedding_dims == 0:
            model.disable_nonsym_embeddings = True
        else:
            model.disable_nonsym_embeddings = False

        logging.info("Built model:")
        print(model)
        return model

    @staticmethod
    def _compute_num_sym_embeddings(args):
        if args.num_sym_embedding_dims is None:
            num_sym_embedding_dims = 100
            if args.max_basket_size != np.inf:
                num_sym_embedding_dims = args.max_basket_size
            return num_sym_embedding_dims
        else:
            return args.num_sym_embedding_dims

    @staticmethod
    def _compute_num_nonsym_embeddings(args):
        if args.num_nonsym_embedding_dims is None:
            num_nonsym_embedding_dims = 10
            if args.num_sym_embedding_dims is not None:
                num_nonsym_embedding_dims = args.num_sym_embedding_dims / 10

            return num_nonsym_embedding_dims
        else:
            return args.num_nonsym_embedding_dims

    @classmethod
    def _load_model(cls, arguments, model, dataset):
        args_dict = arguments.args_dict
        # load or train model
        loaded = None
        try:
            loaded = cls._load_serialized_model(arguments, model)
        except Exception as e:
            logging.error(f"Could not load serialized model due to '{e}'")
        ofile = None
        if loaded is None:
            logging.info("Couldn't load model checkpoint; will retrain")
            # train model
            if model.is_baseline:
                return cls._learn_baseline_model(arguments, model, dataset)
            ofile = cls._learn_dpp_model(arguments, model,
                                         dataset.train_data,
                                         dataset.val_data,
                                         dataset.test_data,
                                         dataset.basket_size_buckets)
            cls._serialize_model(arguments, model)
        else:
            logging.info("Loaded model from checkpoint")
        logging.info("Loaded model:")
        print(model)
        return ofile

    @classmethod
    def _load_serialized_model(cls, arguments, model):
        args = arguments.args
        if not cls._model_can_be_serialized(args):
            return
        persisted_models_path = cls._get_persisted_model_path(args)

        if os.path.exists(persisted_models_path):
            model.load_state_dict(torch.load(persisted_models_path))
            return model

    @classmethod
    def _serialize_model(cls, arguments, model):
        args = arguments.args
        if not cls._model_can_be_serialized(args):
            return

        persisted_model_path = cls._get_persisted_model_path(args)
        head, _ = os.path.split(persisted_model_path)
        if not os.path.exists(head):
            os.makedirs(head)
        torch.save(model.state_dict(),
                   persisted_model_path)

    @staticmethod
    def _model_can_be_serialized(args):
        return args.persisted_model_dir is not None

    @classmethod
    def _get_persisted_model_path(cls, args):
        persisted_model_dir =args.persisted_model_dir
        fname =  cls._persisted_model_fname(args.scores_file).split('/')[-1]

        return os.path.join(persisted_model_dir, fname)

    @staticmethod
    def _persisted_model_fname(scores_file):
        return scores_file + ".torch"

    @staticmethod
    def _learn_baseline_model(arguments, model, dataset):
        return model.do_learning(dataset)

    @staticmethod
    def _learn_dpp_model(arguments, model, train_data, val_data,
                         test_data, basket_size_buckets):
        args_dict = arguments.args_dict
        args = arguments.args
        # train model
        _, ofile = do_learning(model,
                               **{"train_data": train_data,
                                  "val_data": val_data,
                                  "test_data": test_data,
                                  "num_iterations": args.num_iterations,
                                  "alpha_train": arguments.alpha,
                                  "beta_train": arguments.beta,
                                  "disable_eval": arguments.disable_eval,
                                  "inference": arguments.inference,
                                  "learning_rate": arguments.lr,
                                  "eval_freq": 20,
                                  "buckets": basket_size_buckets,
                                  "num_bootstraps": arguments.num_bootstraps,
                               })
        return ofile

# def main():
if __name__ == "__main__":
    # N.B: run script with --help to see command-line options
    logging.info("Process %i: starting" % os.getpid())

    seed = 31
    rng = check_random_state(seed)

    arguments = Args.build_from_cli()
    args = arguments.args
    args_dict = arguments.args_dict
    num_val_baskets = args.num_val_baskets
    num_test_baskets = args.num_test_baskets

    dataset = Dataset(args, seed, rng, num_val_baskets, num_test_baskets)

    model, ofile = Experiment.build(arguments, dataset)

    # get t-SNE embeddings
    if arguments.tsne:
        Experiment.save_tsne_projections(model, ofile)

    if arguments.inference:
        results_df = Experiment.run(model, arguments, dataset, store_inference_scores=True)
        res = Results(args.dataset_name, results_df)
        print(res)

