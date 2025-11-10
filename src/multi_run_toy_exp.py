import model
from model import DBM
import torch
import os
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import hdmm_torch

from vis import data_summary
from hdmm_torch import HDMM


model_path = 'models'
log_dir = './hpc_runs/unsupervised/bernoulli/Mutations-unsupervised-bernoulli-L:2-nh:[4900, 3600]-ySigma:1.0-rho:0.1-nMulti:100-lr:0.005-momentum:0.9-bs:1000-gamma:0.0001-epoch:1000-seed:0-0UAqFzWs'
epoch = 1000
dbm_model = DBM(nv=4801, nh=[4900, 3600], L=2)

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



sample_size = 100
latent_logits_slice = torch.from_numpy(latent_logits_all[:sample_size])
label_slice = torch.from_numpy(y_all[:sample_size])

# Convert logits to probabilities
probs_slice = torch.nn.functional.softmax(latent_logits_slice, dim=1)

# Sample one index per row
M = 5000
N, D = probs_slice.shape

# Sample all at once
samples = torch.multinomial(probs_slice, num_samples=M, replacement=True)
print(samples.shape)

one_hot_samples = torch.nn.functional.one_hot(samples, num_classes=D)  # (N, M, D)

dataset = (one_hot_samples, label_slice)


vocab_size = D
struct_upbd = {"G0": 20, "G1": 10, "G2": 5}
for seed in range(0, 110, 10):
    log_dir = f"./logs/seed_{seed}"
    model = HDMM(struct_upbd, vocab_size=vocab_size, device='cpu', seed=seed)
    print(f"seed {seed} Model initialized.")

    z_gen, z_reg, local_category_assignments, doc_values, log_prob = model.infer(dataset[0].to(torch.float32), reg=dataset[1].to(torch.float32), num_iters=300, sanity_check=False, plot_gap=300, log_dir=log_dir)
    print(f"seed {seed} Inference completed.")

    post_state = {
        "struct_params": model.struct_params,
        "struct_values": model.SV,
        "mixture_components": {
            "generation": model.mixture_components["generation"],
            "regression_mu": model.mixture_components["regression_mu"],
            "regression_sigma": model.mixture_components["regression_sigma"],
        },
        "local_category_assignments": local_category_assignments,
        "doc_values": None,
        "words": {
            "z_gen": z_gen,
            "z_reg": z_reg,
            "obs": dataset[0],
            "reg": dataset[1],
        },
    }
    model_return = (post_state, None)
    data_summary(model_return, {"x": dataset[0], "y": dataset[1]}, struct_upbd, log_dir=log_dir)