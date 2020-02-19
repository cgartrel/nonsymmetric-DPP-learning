"""
"""
import math
import copy
import random
import logging

import torch

torch.manual_seed(1234)
random.seed(1234)

# Noise added to make a singular matrix non-singular
epsilon = 1e-4


class NonSymmetricDPPSampler(object):
    def __init__(self, num_threads=1, device=torch.device("cpu")):
        self.num_threads = num_threads
        self.device = device

    # Generates samples using the Cholesky-based DPP sampling algorithm
    # described at
    # https://dppy.readthedocs.io/en/latest/finite_dpps/exact_sampling.html#finite-dpps-exact-sampling-cholesky-method
    # Works for both symmetric and nonsymmetric DPPs.
    def generate_samples(self, num_samples, dpp=None, V=None, B=None, C=None, L=None):
        samples = []

        if L is None:
            if dpp.disable_nonsym_embeddings and V is None:
                V = dpp.forward(dpp.all_items_in_catalog_set_var)
            elif V is None and B is None and C is None:
                V, B, C = dpp.forward(dpp.all_items_in_catalog_set_var)

            # Symmetric component of kernel
            L = V.mm(V.transpose(0, 1))

            # Nonsymmetric component of kernel
            if not dpp.disable_nonsym_embeddings:
                nonsymm = B.mm(C.transpose(0, 1)) - C.mm(B.transpose(0, 1))
                kernel = L + nonsymm
            else:
                kernel = L
        else:
            kernel = L

        num_items_in_catalog = kernel.size()[0]

        # Convert kernel to marginal kernel (K)
        eye = torch.eye(num_items_in_catalog)
        K = eye - (kernel + eye).inverse()

        for i in range(num_samples):
            K_copy = K.clone().detach()
            sample = []

            for j in range(num_items_in_catalog):
                if torch.rand(1) < K_copy[j, j]:
                    sample.append(j)
                else:
                    K_copy[j, j] -= 1

                K_copy[j + 1:, j] /= K_copy[j, j]
                K_copy[j + 1:, j + 1:] -= torch.ger(K_copy[j + 1:, j], K_copy[j, j + 1:])

            samples.append(sample)

        return samples

    def condition_dpp_on_items_observed(self, model, items_observed,
                                             V=None, B=None, C=None):
        """
          Conditions nonsymmetric DPP on items observed in a set/basket
        """
        all_items_in_catalog_set = model.all_items_in_catalog_set
        if V is None and B is None and C is None:
            if model.disable_nonsym_embeddings:
                V = model.forward(model.all_items_in_catalog_set_var)
                V = V.to(self.device)
            else:
                V, B, C = model.forward(model.all_items_in_catalog_set_var)
                V = V.to(self.device)
                B = B.to(self.device)
                C = C.to(self.device)

        # Symmetric component of kernel
        L = V.mm(V.transpose(0, 1))

        # Nonsymmetric component of kernel
        if not model.disable_nonsym_embeddings:
            nonsymm = B.mm(C.transpose(0, 1)) - C.mm(B.transpose(0, 1))
            kernel = L + nonsymm
        else:
            kernel = L

        items_observed_set = set(items_observed)
        all_items_not_in_observed = list(all_items_in_catalog_set - items_observed_set)

        kernel_items_not_observed = kernel[all_items_not_in_observed, all_items_not_in_observed]
        kernel_items_observed = kernel[items_observed, :][:, items_observed]

        try:
            kernel_conditioned_on_items_observed = kernel_items_not_observed - \
                                                   kernel[all_items_not_in_observed, :][:, items_observed].mm(
                                                       kernel_items_observed.inverse()).mm(
                                                       kernel[items_observed, :][:, all_items_not_in_observed])
        except RuntimeError as e:
            # Add small amount of noise to kernel_items_observed to make it non-singular
            eye = torch.eye(kernel_items_observed.size()[0]).to(self.device)
            kernel_items_observed += eye * epsilon
            kernel_conditioned_on_items_observed = kernel_items_not_observed - \
                                                   kernel[all_items_not_in_observed, :][:, items_observed].mm(
                                                       kernel_items_observed.inverse()).mm(
                                                       kernel[items_observed, :][:, all_items_not_in_observed])

        # Create a dict mapping original itemsIds in all_items_not_in_observed to row/col
        # indices in K_conditioned_on_items_observed (marginal kernel matrix, K, conditioned on items observed)
        item_ids_to_K_conditioned_on_items_observed_row_col_indices = {}
        k_matrix_items_observed_row_index = 0
        for item_id in all_items_not_in_observed:
            item_ids_to_K_conditioned_on_items_observed_row_col_indices[item_id] = k_matrix_items_observed_row_index

            k_matrix_items_observed_row_index += 1

        return kernel_conditioned_on_items_observed, \
               item_ids_to_K_conditioned_on_items_observed_row_col_indices

    # Computes the unnormalized probabilities of observing each item in next_items, given the nonsymmetric low-rank DPP
    # with the specified embedding matrices, and a set of observed items.  A conditional k-DPP is used to compute these
    # probabilities.
    # @profile
    def compute_next_item_probs_conditional_kdpp(
            self, model, items_observed, next_items,
            return_next_item_probs_as_dict=True, V=None, B=None, C=None):

        kernel_conditioned_on_items_observed, \
        item_ids_to_K_conditioned_on_items_observed_row_col_indices = self.condition_dpp_on_items_observed(
            model, items_observed, V=V, B=B, C=C)

        next_item_probs_vec = torch.empty(len(next_items))
        if return_next_item_probs_as_dict:
            next_item_probs = dict.fromkeys(next_items)
            for next_item_id in next_items:
                items_observed_row_col_index = item_ids_to_K_conditioned_on_items_observed_row_col_indices[next_item_id]

                # The conditional probability for a singleton item is simply the corresponding
                # diagonal element of kernel_conditioned_on_items_observed
                next_item_probs[next_item_id] = kernel_conditioned_on_items_observed[items_observed_row_col_index,
                                                                                     items_observed_row_col_index]
                next_item_probs_vec = kernel_conditioned_on_items_observed[items_observed_row_col_index,
                                                                           items_observed_row_col_index]

            return next_item_probs
        else:
            # Avoiding creation of next_item_probs dict reduces runtime of
            # compute_next_item_probs_conditional_kdpp() by about 40 - 50%
            return next_item_probs_vec, item_ids_to_K_conditioned_on_items_observed_row_col_indices

