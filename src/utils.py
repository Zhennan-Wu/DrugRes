import random
import string
import torch
from torchvision.utils import make_grid
from torchvision.io import write_jpeg
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def int2bit(x, bits=8):
    mask = 2**torch.arange(bits, device=x.device)
    out = x.unsqueeze(-1).bitwise_and(mask).ne(0)

    return out

def bit2int(x, bits=8):
    mask = 2 ** torch.arange(bits, device=x.device)
    out = torch.sum(mask * x, -1)

    return out

def float2bit(x, bits=8):
    out = x.mul(2**bits - 1).int()
    out = int2bit(out, bits)

    return out

def bit2float(x, bits=8):
    out = bit2int(x, bits).float()
    out /= 2**bits - 1

    return out

def binarize(x):
    return (x > 0.5).float()

def generate_id(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def visualize_curve(energy_mean, energy_std, loss_mean, loss_std, epoch, log_dir, writer=None):
    epochs = np.arange(1, len(energy_mean) + 1)

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.set_title("Energy and Loss Curves", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # Plot energy curve with shaded std
    ax.plot(epochs, energy_mean, color='tab:blue', label='Energy', linewidth=2)
    ax.fill_between(epochs,
                    energy_mean - energy_std,
                    energy_mean + energy_std,
                    color='tab:blue',
                    alpha=0.2)

    # Plot loss curve with shaded std
    ax.plot(epochs, loss_mean, color='tab:orange', label='Loss', linewidth=2)
    ax.fill_between(epochs,
                    loss_mean - loss_std,
                    loss_mean + loss_std,
                    color='tab:orange',
                    alpha=0.2)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()

    # --- Save as image ---
    save_path = os.path.join(log_dir, f"energy_curve_epoch_{epoch:04d}.png")
    fig.savefig(save_path, dpi=300)

    # --- Log to TensorBoard ---
    if writer is not None:
        writer.add_figure("Energy_Loss_Curves", fig, global_step=epoch)

    plt.close(fig)


def transfer_state_to_data(state, struct_upbd):
    """
    Given the HDMM state, extract the mixture components and category assignments
    in a format similar to the ground truth data for evaluation.

    Args:
        state: dict, the HDMM state returned by the model function or Gibbs sampler.
        struct_upbd: dict, structure upper bounds.
    Returns:
        data: dict, containing 'word_dists' and 'category_assignments'.
    """
    N = state["words"]["obs"].shape[0]
    data = {}
    data["word_dists"] = state["mixture_components"]["generation"].detach().cpu().numpy()
    data["reg_means"] = state["mixture_components"]["regression_mu"].detach().cpu().numpy()
    data["reg_std"] = state["mixture_components"]["regression_sigma"].detach().cpu().numpy()
    data["super_labels"], data["base_labels"] = transfer_hierarchy_to_data_labels(state["local_category_assignments"].detach().cpu().numpy(), [struct_upbd["G1"], struct_upbd["G2"]])
    data["x_labels"] = state["words"]["z_gen"].detach().cpu().numpy()
    data["y_labels"] = state["words"]["z_reg"].detach().cpu().numpy()
    data["x"] = np.array(state["words"]["obs"])
    data["y"] = np.array(state["words"]["reg"])

    # Extract super-cluster mixture weights
    G1 = struct_upbd["G1"]
    G2 = struct_upbd["G2"]
    S = G1
    C = G2

    super_weights = state["struct_values"]["G1"].detach().cpu().numpy()
    assert super_weights.shape == (S, struct_upbd["G0"])
    data["super_mix_weights"] = super_weights

    base_weights = np.transpose(state["struct_values"]["G2"].detach().cpu().numpy(), (1, 0, 2))  # (S, C, K)
    assert base_weights.shape == (S, C, struct_upbd["G0"])
    data["child_mix_weights"] = base_weights

    return data


def transfer_hierarchy_to_data_labels(local_cats: np.ndarray, level_dims: list[int]) -> np.ndarray:
    """
    Reverse of transfer_data_labels_to_hierarchy.
    Given local_cats and cluster sizes per level, reconstruct absolute labels.

    Args:
        local_cats: (N, L) array of local indices
        level_dims: list of ints, max cluster size per level

    Returns:
        data_labels: (N, L) array of absolute indices per level
    """
    N, L = local_cats.shape
    data_labels = np.zeros((N, L), dtype=np.int32)

    # level 0: absolute = local (super labels)
    data_labels[:, 0] = local_cats[:, 0]

    # deeper levels: absolute id = parent_abs * K[level] + local_id
    for level in range(1, L):
        parent_abs = data_labels[:, level - 1]
        data_labels[:, level] = parent_abs * level_dims[level] + local_cats[:, level]
    
    labels_per_level = [data_labels[:, i] for i in range(L)]

    return labels_per_level
