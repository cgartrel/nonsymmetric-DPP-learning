"""
"""
import random

import torch
import copy
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from nonsymmetric_dpp_sampling import NonSymmetricDPPSampler
from utils import LogLikelihood

torch.manual_seed(1234)
random.seed(1234)

# Determine whether to use GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_auc_from_logliks(pos_pred_logliks, random_logliks):
    labels = np.array([1] * len(pos_pred_logliks) + [0] * len(random_logliks))
    predictions = np.array(pos_pred_logliks + random_logliks).ravel()
    return roc_auc_score(labels, predictions)


class NonSymmetricDPPPrediction(nn.Module):
    is_baseline = False

    def __init__(self, num_threads=1):
        super(NonSymmetricDPPPrediction, self).__init__()
        self.num_threads = num_threads
        self.dpp_sampler = NonSymmetricDPPSampler(num_threads=num_threads)
        torch.set_num_threads(num_threads)

    def get_basket_completion(self, basket, V=None, B=None, C=None):
        basket = copy.copy(basket)
        last = basket.pop()

        next_items = np.setdiff1d(np.arange(self.item_catalog_size), basket)
        next_item_probs_vec = self.dpp_sampler.condition_dpp_on_items_observed_greedy(
            self, items_observed=basket, V=V, B=B, C=C).detach().cpu()
        prediction = np.zeros(self.item_catalog_size)
        prediction[next_items] = next_item_probs_vec
        return prediction, last, basket

    def get_predictions(self, test_data, V=None, B=None, C=None):
        if V is None and B is None and C is None:
            if self.disable_nonsym_embeddings:
                V = self.forward(self.all_items_in_catalog_set_var)
            else:
                V, B, D = self.forward(self.all_items_in_catalog_set_var)
                C = D - D.transpose(0, 1)
        for i, basket in tqdm(enumerate(test_data)):
            if len(basket) > 1:
                yield self.get_basket_completion(basket, V=V, B=B, C=C)

    def get_MPR_results(self, test_data):
        predictions, targets, batch_inputs = self.get_predictions(test_data)
        for target in targets:
            for i in target:
                assert i < self.item_catalog_size
        preds_sorted = [
            1 - len(np.where(predictions[i] > predictions[i][int(targets[i])])[0]) / 
            float(sum(predictions[i] != 0.0)) for i in range(len(batch_inputs))]

        MPR_sorted = 100 * np.mean(np.array(preds_sorted))
        return MPR_sorted

    def get_prec_at_k(self, k, test_data):
        predictions, targets, batch_inputs = self.get_predictions(
            test_data)

        preds_top_k = [1 if targets[i] in list(np.argpartition(
            -predictions[i], k)[:k]) else 0 for i in range(len(batch_inputs))]
        prec_at_k = 100 * np.count_nonzero(
            np.array(preds_top_k)) / len(batch_inputs)
        return prec_at_k

    def _get_pre_mpr(self, prediction, target):
        return 1 - len(np.where(
            prediction >= float(prediction[int(target)]))[0]) / float(sum(prediction != 0.0))

    def get_MPR_results_for_predictions(self, predictions, targets,
                                        batch_inputs):
        preds_sorted = [
            1 - len(np.where(predictions[i] > predictions[i][int(targets[i])])[0]) 
            / float(sum(predictions[i] != 0.0)) for i in range(len(batch_inputs))]

        MPR_sorted = 100 * np.mean(np.array(preds_sorted))
        return MPR_sorted

    def _get_top_k(self, k, prediction, target, weight=None):
        if weight is None:
            weight = 1
        return weight if target in list(np.argpartition(-prediction,
                                                        k)[:k]) else 0

    def get_prec_at_k_for_predictions(self, k, predictions, targets,
                                      batch_inputs, weights=None):
        # XXX weighting scheme doesn't yet seem to work well yet!!!
        if weights is not None:
            # Normalize item weights so that they sum to one across the item catalog
            sum_weights = sum(weights)
            weights = weights / sum_weights

            weights = [weights[targets[i]] for i in range(len(batch_inputs))]
        preds_top_k = [(1 if weights is None else weights[i])
                             if targets[i] in list(np.argpartition(-predictions[i],
                                                                   k)[:k]) else 0
                             for i in range(len(batch_inputs))]
        if weights is None:
            prec_at_k = np.mean(preds_top_k)
        else:
            prec_at_k = np.sum(preds_top_k) / np.sum(weights)
        return 100. * prec_at_k

    def generate_random_sets(self, test_data, list_of_set_sizes, item_catalog_size):
        products = np.arange(item_catalog_size)
        sets = list()
        for size in list_of_set_sizes:
            sets.append(np.random.choice(products, size=size, replace=False))
        return sets

    def get_AUC_results(self, test_data):
        # true baskets
        positive_preds = LogLikelihood.compute_log_likelihood(
            self, test_data, reduce=False).data.numpy()

        # random baskets
        random_negatives = self.generate_random_sets(
            test_data, [len(item) for item in test_data],
            self.item_catalog_size)
        random_preds = LogLikelihood.compute_log_likelihood(random_negatives,
                                                            reduce=False).data.numpy()

        # compute AUC for the problem of distinguishing between DPP-sampled and
        # randomly generated baskets
        labels = np.append(np.ones(len(positive_preds), dtype=np.int64),
                           np.zeros(len(random_preds), dtype=np.int64))
        predictions = np.append(positive_preds, random_preds)
        auc_score = roc_auc_score(labels, predictions)
        return auc_score

    def _get_AUC_results_for_data(self, test_data, negatives=None):
        if negatives is None or True:
            negatives = self.generate_random_sets(
                test_data, [len(item) for item in test_data],
                self.item_catalog_size)
        else:
            negatives = np.random.choice(negatives, len(test_data), replace=False)

        pos_pred_logliks = LogLikelihood.compute_log_likelihood(
            self, test_data, mapped=False, alpha_regularization=0,
            reduce=False)
        pos_pred_logliks = [loglik.item() for loglik in pos_pred_logliks]

        random_logliks = LogLikelihood.compute_log_likelihood(
            self, negatives, alpha_regularization=0, reduce=False)
        random_logliks = [loglik.item() for loglik in random_logliks]
        return pos_pred_logliks, random_logliks

    def get_AUC_results_for_data(self, data, return_raw=False, negatives=None):
        pos_pred_logliks, random_logliks = self._get_AUC_results_for_data(data,
                                                                          negatives=negatives)
        if return_raw:
            return pos_pred_logliks, random_logliks
        else:
            return get_auc_from_logliks(pos_pred_logliks, random_logliks)
