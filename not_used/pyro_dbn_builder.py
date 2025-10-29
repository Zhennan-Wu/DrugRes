import numpy as np
from pyro_bernoulli_rbm import BernoulliRBM
from pyro_gaussian_rbm import GaussianRBM
from pyro_bernoulli_softmax_rbm import BernoulliSoftmaxRBM
from pyro_gaussian_softmax_rbm import GaussianSoftmaxRBM
from pyro_mixed_rbm import MixedRBM
from pyro_mixed_softmax_rbm import MixedSoftmaxRBM
import torch
import pyro
import pyro.distributions as dist

    
def generate_rbm_types(input_type, intermediate_type, latent_type, rbm_layer_sizes):
    rbm_types = [intermediate_type]*len(rbm_layer_sizes)
    rbm_types[0] = input_type
    rbm_types[-1] = latent_type
    return rbm_types


def check_model(rbm_layer_sizes):
    layer_descending = all(isinstance(x, int) for x in rbm_layer_sizes) and all(x >= y for x, y in zip(rbm_layer_sizes, rbm_layer_sizes[1:]))
    if not layer_descending:
        raise ValueError("rbm layers dimensions are not in descending order")
    if (len(rbm_layer_sizes) <= 1):
        raise ValueError("dbn need to have more than 1 layer")
    

def get_rbm_factory(rbm_type, **global_params):
    """
    Returns a factory that produces an RBM instance based on latent_type.
    global_params can include 'sigma' (for GaussianRBM) or others.
    """
    rbm_map = {
        'bernoulli': BernoulliRBM,
        'softmax': BernoulliSoftmaxRBM,
        'gaussian': GaussianRBM,
        'gaussian_softmax': GaussianSoftmaxRBM,
        'mixed': MixedRBM,
        'mixed_softmax': MixedSoftmaxRBM,
    }

    if rbm_type not in rbm_map:
        raise ValueError(f"Unsupported latent_type: {rbm_type}. Available: {list(rbm_map.keys())}")
    
    RBMClass = rbm_map[rbm_type]

    def factory(**rbm_params):
        if 'bernoulli' not in rbm_type:
            if 'sigma' not in global_params:
                raise ValueError("GaussianRBM requires 'sigma' to be specified")
            return RBMClass(sigma=global_params['sigma'], **rbm_params)
        else:
            return RBMClass(**rbm_params)

    return factory


def build_unsupervised_dbn(rbm_layer_sizes: list=[512, 256], input_type: str="bernoulli", intermediate_type: str="bernoulli", latent_type: str="bernoulli", sigma: float=1.0, learning_rate: float=0.06, grad_max: float=1e6):
    """
    Create a stacked RBM-based Deep Belief Network using sklearn Pipeline.

    Parameters:
        rbm_layer_sizes (list of int): List of hidden units per RBM layer.
        learning_rate (float): Learning rate for all RBMs.
        n_iter (int): Number of training iterations per RBM.
        verbose (bool): Whether to print training logs.
        random_state (int): Seed for reproducibility.

    Returns:
        dbn_pipeline (Pipeline): A scikit-learn Pipeline of stacked RBMs.
    """
    check_model(rbm_layer_sizes)
    rbm_types = generate_rbm_types(input_type, intermediate_type, latent_type, rbm_layer_sizes)
    steps = []

    for i, hidden_dim in enumerate(rbm_layer_sizes):
        rbm_factory = get_rbm_factory(rbm_types[i], sigma=sigma)
        rbm = rbm_factory(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            grad_max=grad_max,
        )
        steps.append((f'{rbm_types[i]}_rbm_{i+1}', rbm))

    return steps

