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