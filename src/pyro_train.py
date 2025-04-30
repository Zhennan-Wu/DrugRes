import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import binarize
import torch
import pyro
import pyro.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from umap import UMAP
from utils import visualize_data


def fit_dataloader(dataloader, model_pipeline, epochs, batch_size, gaussian_indices=None, savefile="dbn_", showplot=True):
    """
    Fit each RBM layer in a DBN pipeline using batch-wise training from a PyTorch DataLoader.
    """
    input_loader = dataloader

    # Access RBM layers from the pipeline
    rbm_layers = [step[1] for step in model_pipeline]
    rbm_types =  [step[0] for step in model_pipeline]

    # Train each layer one by one
    for layer_idx, rbm in enumerate(rbm_layers):
        print(f"\nTraining RBM Layer {layer_idx+1} {rbm_types[layer_idx]} with {rbm.n_components} hidden units")
        if "bernoulli" in rbm_types[layer_idx]:
            input_data = binarize(input_data, input_data.mean())
            "Input data binarized"

        rbm.batch_fit(input_loader, epochs, gaussian_indices)

        latent_vars = []
        labels = []
        for X, y in input_loader:
            latent_vars.append(rbm.transform(X).detach().cpu())
            labels.append(y)
        latent_vars = torch.cat(latent_vars, dim=0)
        labels = torch.cat(labels, dim=0)
        latent_dataset = TensorDataset(latent_vars, labels)
        input_loader = DataLoader(latent_dataset, batch_size, shuffle=False)

        visualize_data(input_loader, layer_idx, savefile, showplot)

