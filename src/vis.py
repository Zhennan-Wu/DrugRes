from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap
import os
import jax.numpy as jnp
import jax
import torch
from utils import transfer_state_to_data


def data_summary(model_return, data, struct_upbd, log_dir=None):
    N = data["x"].shape[0]
    
    model_data = transfer_state_to_data(model_return[0], struct_upbd)
    tsne_visualization(model_data, struct_upbd, log_dir)
    umap_visualization(model_data, struct_upbd, log_dir)
    

def tsne_visualization(data, struct_upbd, log_dir=None):
    x_agg = data["x"].sum(axis=1) # (N, V)
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    x_embedded = tsne.fit_transform(x_agg)
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # Plot t-SNE colored by super categories
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=data["super_labels"], cmap="tab10", s=10)
    plt.colorbar(scatter, ticks=range(6))
    plt.title("t-SNE of aggregated x colored by super categories")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")

    # Plot t-SNE colored by base categories
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=data["base_labels"], cmap="tab20", s=10)
    plt.colorbar(scatter, ticks=range(18))
    plt.title("t-SNE of aggregated x colored by base categories")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    if log_dir is not None:
        plt.savefig(f"{log_dir}/tsne_visualization.png", dpi=300)
    else:
        plt.show()

    # Plot distribution of y by super categories
    plt.figure(figsize=(8, 4))
    for super_id in range(struct_upbd["G1"]):
        y_vals = data["y"][data["super_labels"] == super_id]
        plt.hist(y_vals, bins=20, alpha=0.5, label=f"Super {super_id}")
    plt.title("Distribution of y by Super Categories")
    plt.xlabel("y")
    plt.ylabel("Count")
    plt.legend()
    if log_dir is not None:
        plt.savefig(f"{log_dir}/y_distribution_super.png", dpi=300)
    else:   
        plt.show()

    plt.figure(figsize=(8, 4))
    for super_id in range(struct_upbd["G1"]):
        for base_offset in range(struct_upbd["G2"]):
            base_id = super_id * struct_upbd["G2"] + base_offset
            y_vals = data["y"][data["base_labels"] == base_id]
            plt.hist(y_vals, bins=20, alpha=0.5, label=f"Base {base_id}")

    plt.title("Distribution of y by Base Categories")
    plt.xlabel("y")
    plt.ylabel("Count")
    plt.legend()
    if log_dir is not None:
        plt.savefig(f"{log_dir}/y_distribution_base.png", dpi=300)
    else:
        plt.show()


def umap_visualization(data, struct_upbd, log_dir=None):
    x_agg = data["x"].sum(axis=1)
    # === UMAP projection ===
    reducer = umap.UMAP(random_state=0)
    x_umap = reducer.fit_transform(x_agg)
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # === UMAP Plots ===
    plt.figure(figsize=(10, 4))

    # Super category coloring
    plt.subplot(1, 2, 1)
    plt.scatter(x_umap[:, 0], x_umap[:, 1], c=data["super_labels"], cmap="tab10", s=10)
    plt.title("UMAP by Super Category")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # Base category coloring
    plt.subplot(1, 2, 2)
    plt.scatter(x_umap[:, 0], x_umap[:, 1], c=data["base_labels"], cmap="tab20", s=10)
    plt.title("UMAP by Base Category")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    if log_dir is not None:
        plt.savefig(f"{log_dir}/umap_visualization.png", dpi=300)
    else:
        plt.show()

    # === Mixture Component Word Distributions ===
    num_components = struct_upbd["G0"]
    V = data["x"].shape[-1]
    word_dists = data["word_dists"]
    super_mix_weights = data["super_mix_weights"]
    child_mix_weights = data["child_mix_weights"]

    # Bar plots for each component
    fig, axs = plt.subplots(struct_upbd["G0"], 1, figsize=(20, 20))
    for i in range(num_components):
        ax = axs[i]
        ax.bar(range(V), word_dists[i])
        ax.set_title(f"Component {i}")
        ax.set_xlabel("Word ID")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, word_dists.max().item() * 1.1)

    fig.suptitle("Word Distributions of 10 Shared Mixture Components", fontsize=16)
    if log_dir is not None:
        plt.savefig(f"{log_dir}/word_distributions.png", dpi=300)
    else:
        plt.show()

    # Bar plots for each component
    fig, axs = plt.subplots(struct_upbd["G1"], 1, figsize=(20, 6))
    for i in range(struct_upbd["G1"]):
        ax = axs[i]
        ax.bar(range(struct_upbd["G0"]), super_mix_weights[i])
        ax.set_title(f"Super Category {i}")
        ax.set_ylabel("Weights")
        ax.set_ylim(0, super_mix_weights.max().item() * 1.1)

    fig.suptitle("Super Category Weights of 10 Shared Mixture Components", fontsize=16)
    if log_dir is not None:
        plt.savefig(f"{log_dir}/super_category_weights.png", dpi=300)
    else:
        plt.show()

    # Bar plots for each component
    fig, axs = plt.subplots(struct_upbd["G1"]*struct_upbd["G2"], 1, figsize=(20, 20))
    for i in range(struct_upbd["G1"]):
        for j in range(struct_upbd["G2"]):
            ax = axs[i * struct_upbd["G2"] + j]
            ax.bar(range(struct_upbd["G0"]), child_mix_weights[i][j])
            ax.set_title(f"Super Category {i} Child Category {j}")
            ax.set_ylabel("Weights")
            ax.set_ylim(0, child_mix_weights.max().item() * 1.1)

    fig.suptitle("Child Category Weights of 10 Shared Mixture Components", fontsize=16)
    if log_dir is not None:
        plt.savefig(f"{log_dir}/child_category_weights.png", dpi=300)
    else:
        plt.show()


def likelihood_visualization(likelihood_mean, likelihood_std, epoch=0, log_dir=None):
    if isinstance(likelihood_mean, torch.Tensor):
        likelihood_mean = likelihood_mean.detach().cpu().numpy()
    if isinstance(likelihood_std, torch.Tensor):
        likelihood_std = likelihood_std.detach().cpu().numpy()
    iterations = np.arange(1, len(likelihood_mean) + 1)
    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.set_title("Likelihood Curves", fontsize=14, fontweight='bold')
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Likelihood", fontsize=12)

    # Plot likelihood curve with shaded std
    ax.plot(iterations, likelihood_mean, color='tab:orange', label='Likelihood', linewidth=2)
    ax.fill_between(iterations,
                    likelihood_mean - likelihood_std,
                    likelihood_mean + likelihood_std,
                    color='tab:orange',
                    alpha=0.2)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()

    # --- Save as image ---
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_path = os.path.join(log_dir, f"likelihood_curve_epoch_{epoch:04d}.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def hdmm_visualization(model, file_prefix):
    for mix_idx, word_dist in enumerate(model.mixture_components["generation"]):
        plt.figure(figsize=(6, 4), dpi=150)
        plt.bar(range(word_dist.shape[0]), word_dist)
        plt.title(f"Mixture Distribution Number {mix_idx}")
        plt.xlabel("Word ID")
        plt.ylabel("Weights")
        plt.ylim(0, word_dist.max().item() * 1.1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{file_prefix}_hdmm_word_distribution_{mix_idx}.png")
        plt.close()
    
    for reg_idx, mu, sigma in zip(np.arange(model.K), model.mixture_components["regression_mu"], model.mixture_components["regression_std"]):
        plt.figure(figsize=(6, 4), dpi=150)
        # generate x range
        x = jnp.linspace(mu - 4*sigma, mu + 4*sigma, 400)

        # pdf of normal distribution
        pdf = jax.scipy.stats.norm.pdf(x, loc=mu, scale=sigma)

        # plot
        plt.figure(figsize=(6,4))
        plt.plot(x, pdf, lw=2, label=fr'$\mathcal{{N}}({mu}, {sigma}^2)$')
        plt.title("Regression Component Distribution Number {}".format(reg_idx))
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{file_prefix}_hdmm_regression_distribution_{reg_idx}.png")
        plt.close()
    
    for depth in range(len(model.cluster_dims)):
        hierarchy_shape = model.cluster_dims[:depth+1]
        grids = jnp.meshgrid(*[jnp.arange(n) for n in hierarchy_shape], indexing="ij")
        hierarchy_cats = jnp.stack([g.ravel() for g in grids], axis=-1)
        for cat in hierarchy_cats:
            rev_idx = tuple(cat.tolist()[::-1])
            plt.figure(figsize=(6, 4), dpi=150)
            mix_weights = model.struct_value[f"G{depth}"][rev_idx]
            plt.bar(range(mix_weights.shape[0]), mix_weights)
            plt.title(f"Mixture Weights for Category Hierarchy {cat.tolist()}")
            plt.xlabel("Mixture Component Index")
            plt.ylabel("Weight")
            plt.ylim(0, mix_weights.max().item() * 1.1)
            plt.legend()
            plt.tight_layout()
            cat_str = "_".join([str(c) for c in cat.tolist()])
            plt.savefig(f"{file_prefix}_hdmm_mixture_weights_category_hierarchy_{cat_str}.png")
            plt.close()
