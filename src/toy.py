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
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import importlib
from hdmm import HDMM
import pickle
from vis import likelihood_visualization


if __name__ == "__main__":
    # log_dir = './runs/Mutations-bits:1-L:2-nh:[3600, 2500]-lr:0.01-momentum:0-bs:1000-gamma:0.001-epoch:1000-seed:0-0UAqFzWs'
    log_dir = './runs/Mutations-bits:1-L:2-nh:[4900, 3600]-lr:0.005-momentum:0.9-bs:1000-gamma:0.0001-epoch:1000-seed:0-p2BZQXg2'
    model_path = 'models'
    # epoch = 360
    epoch = 220
    # dbm_model = DBM(size=1, nc=4801, nh=[3600, 2500], bits=1, L=2)
    dbm_model = DBM(size=1, nc=4801, nh=[4900, 3600], bits=1, L=2)

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

    hdmm_model = HDMM(struct_upbd={"G0": 20, "G1": 6, "G2": 9}, vocab_size=3600)
    M = 100

    data = torch.load('cell_drug_response_samples.pt')
    X = data['X'][:5000]
    y = data['y'][:5000]
    data_size = X.shape[0]
    print(f"Data size: {data_size}")

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    h_mode_list = []
    h_rand_list = []
    latent_logits_list = []

    y_list = []
    i = -1
    key = jax.random.PRNGKey(0)
    for epoch in range(100):
        print(f"Epoch {epoch} starting inference...")
        for x_batch, y_batch in dataloader:
            i += 1
            with torch.no_grad():
                h_mode, h_rand, latent_logits = dbm_model.encode(x_batch)
            h_mode = h_mode.detach().numpy()
            h_rand = h_rand.detach().numpy()
            latent_logits = latent_logits.detach().numpy()
            y_batch = y_batch.detach().numpy()
            prob = jax.nn.softmax(jnp.array(latent_logits), axis=1)
            N, D = prob.shape
            logits_expanded = jnp.repeat(prob[:, None, :], M, axis=1)
            logits_flat = logits_expanded.reshape(-1, D)  # (N*M, D)

            # Sample all at once
            key, subkey = jax.random.split(key)
            samples_flat = jax.random.categorical(subkey, logits_flat, axis=-1)  # (N*M,)
            samples = samples_flat.reshape(N, M)
            one_hot_samples = jax.nn.one_hot(samples, num_classes=D)  # (N, M, D)

            labels = jnp.array(y_batch)

            z_gen, z_reg, local_category_assignments, mc, doc_values, log_prob = hdmm_model.infer(one_hot_samples, labels, num_iters=300, key=key, epoch=epoch, datasize=data_size)
            print(f"Batch {i} inference completed.")
            likelihood_visualization(log_prob, np.zeros_like(log_prob), epoch=i, log_dir=f"./toy_results/epoch_{epoch}/")
            state = {
                "struct_params": hdmm_model.struct_params,
                "mixture_components": hdmm_model.mixture_components,
                "struct_values": hdmm_model.struct_values,
            }
            model_dir = "./toy_model_states/epoch_" + str(epoch)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            with open(model_dir + f"/hdmm_model_state_after_batch_{i}.pkl", "wb") as f:
                pickle.dump(state, f)
            

    # # Load
    # with open("jax_model.pkl", "rb") as f:
    #     loaded = pickle.load(f)