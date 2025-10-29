import coderdata as cd
import os
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # Correct import for new RDKit
import pubchempy as pcp
import coderdata_processing
import importlib
import torch
from torch.utils.data import TensorDataset
from model import DBM
from torch.utils.data import DataLoader, RandomSampler
import jax
import jax.numpy as jnp
import jax_hdmm
import importlib
importlib.reload(jax_hdmm)
from jax_hdmm import hdp_model, gibbs_sampler, data_summary
import pickle





if __name__ == "__main__":
    log_dir = './runs/Mutations-bits:1-L:2-nh:[3600, 2500]-lr:0.01-momentum:0-bs:1000-gamma:0.001-epoch:1000-seed:0-0UAqFzWs'
    model_path = 'models'
    epoch = 360
    dbm_model = DBM(size=1, nc=4801, nh=[3600, 2500], bits=1, L=2)

    checkpoint = torch.load(os.path.join(log_dir, model_path, f"model-{epoch}.pt"))
    try:
        dbm_model.load_state_dict(checkpoint)
    except RuntimeError as e:
        if 'module.' in list(checkpoint.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint.items())
            dbm_model.load_state_dict(new_state_dict)
        else:
            raise e
    dbm_model.eval()

    data = torch.load('cell_drug_response_samples.pt')
    X = data['X']
    y = data['y']

    dataset = torch.utils.data.TensorDataset(data['X'], data['y'])
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    h_mode_list = []
    h_rand_list = []
    latent_logits_list = []

    y_list = []
    for x_batch, y_batch in dataloader:
        with torch.no_grad():
            h_mode, h_rand, latent_logits = dbm_model.encode(x_batch)
        h_mode_list.append(h_mode.numpy())
        h_rand_list.append(h_rand.numpy())
        latent_logits_list.append(latent_logits.numpy())
        y_list.append(y_batch.numpy())
    h_mode_all = np.vstack(h_mode_list)
    h_rand_all = np.vstack(h_rand_list)
    latent_logits_all = np.vstack(latent_logits_list)
    y_all = np.hstack(y_list)

    # Convert logits to probabilities
    probs = jax.nn.softmax(jnp.array(latent_logits_all), axis=1)
    key = jax.random.PRNGKey(0)
    sample_size = 500
    probs_slice = probs[:sample_size]
    label_slice = jnp.array(y_all)[:sample_size]

    # Sample one index per row
    M = 200
    N, D = probs_slice.shape
    logits_expanded = jnp.repeat(probs_slice[:, None, :], M, axis=1)

    logits_flat = logits_expanded.reshape(-1, D)  # (N*M, D)

    # Sample all at once
    key, subkey = jax.random.split(key)
    samples_flat = jax.random.categorical(subkey, logits_flat, axis=-1)  # (N*M,)
    samples = samples_flat.reshape(N, M)
    one_hot_samples = jax.nn.one_hot(samples, num_classes=D)  # (N, M, D)

    dataset = (one_hot_samples, label_slice)

    vocab_size = D
    struct_upbd = {"G0": 20, "G1": 5, "G2": 3}

    hdmm14 = hdp_model(dataset, struct_upbd=struct_upbd, vocab_size=vocab_size, seed=60, known_base=False, known_super=False, gen_mixture=None, device="cpu")
    model_return14 = gibbs_sampler(jax.random.PRNGKey(0), hdmm14, struct_upbd, vocab_size, num_iters=100, gt=None, file_prefix="hdmm14", known_base=False, known_super=False, gen_ground_truth=False)
    data_summary(model_return14, data, struct_upbd, file_prefix="hdmm14")


    # Save
    with open("jax_model.pkl", "wb") as f:
        pickle.dump(model_return14, f)

    # # Load
    # with open("jax_model.pkl", "rb") as f:
    #     loaded = pickle.load(f)