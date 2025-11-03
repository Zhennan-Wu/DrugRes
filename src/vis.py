from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap
import os
import jax.numpy as jnp
import jax


def tsne_visualization(data, struct_upbd, file_prefix):
    x_agg = data["x"].sum(axis=1)
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    x_embedded = tsne.fit_transform(x_agg)

    # --- Plot 1: super categories ---
    plt.figure(figsize=(6, 5), dpi=150)
    scatter = plt.scatter(
        x_embedded[:, 0],
        x_embedded[:, 1],
        c=data["super_labels"],
        cmap="tab10",
        s=10
    )
    plt.colorbar(scatter, ticks=range(6))
    plt.title("t-SNE of aggregated x colored by super categories")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_tsne_super.png")
    plt.close()

    # --- Plot 2: base categories ---
    plt.figure(figsize=(6, 5), dpi=150)
    scatter = plt.scatter(
        x_embedded[:, 0],
        x_embedded[:, 1],
        c=data["base_labels"],
        cmap="tab20",
        s=10
    )
    plt.colorbar(scatter, ticks=range(18))
    plt.title("t-SNE of aggregated x colored by base categories")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_tsne_base.png")
    plt.close()

    # Plot distribution of y by super categories
    plt.figure()
    for super_id in range(struct_upbd["G1"]*struct_upbd["G2"]):
        y_vals = data["y"][data["super_labels"] == super_id]
        plt.hist(y_vals, bins=20, alpha=0.5, label=f"Super {super_id}")

    plt.title("Distribution of y by Super Categories")
    plt.xlabel("y")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"{file_prefix}_y_distribution_by_super.png")
    plt.close()


def umap_visualization(data, struct_upbd, file_prefix):
    x_agg = data["x"].sum(axis=1)
    # === UMAP projection ===
    reducer = umap.UMAP(random_state=0)
    x_umap = reducer.fit_transform(x_agg)


    # === UMAP Plots ===

    # --- Plot 1: UMAP by Super Category ---
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(
        x_umap[:, 0],
        x_umap[:, 1],
        c=data["super_labels"],
        cmap="tab10",
        s=10
    )
    plt.title("UMAP by Super Category")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_umap_super.png")
    plt.close()

    # --- Plot 2: UMAP by Base Category ---
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(
        x_umap[:, 0],
        x_umap[:, 1],
        c=data["base_labels"],
        cmap="tab20",
        s=10
    )
    plt.title("UMAP by Base Category")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_umap_base.png")
    plt.close()


    # === Mixture Component Word Distributions ===
    num_components = struct_upbd["G0"]
    V = data["x"].shape[-1]
    word_dists = data["word_dists"]
    super_mix_weights = data["super_mix_weights"]
    child_mix_weights = data["child_mix_weights"]

    # === Bar plots for each component ===
    for i in range(num_components):
        plt.figure(figsize=(6, 4), dpi=150)
        plt.bar(range(V), word_dists[i])
        plt.title(f"Component {i}")
        plt.xlabel("Word ID")
        plt.ylabel("Probability")
        plt.ylim(0, word_dists.max().item() * 1.1)
        plt.tight_layout()
        plt.savefig(f"{file_prefix}_word_distribution_component_{i}.png")
        plt.close()


    # === Bar plots for each super category ===
    for i in range(struct_upbd["G1"]):
        plt.figure(figsize=(6, 4), dpi=150)
        plt.bar(range(struct_upbd["G0"]), super_mix_weights[i])
        plt.title(f"Super Category {i}")
        plt.xlabel("Mixture Components")
        plt.ylabel("Weights")
        plt.ylim(0, super_mix_weights.max().item() * 1.1)
        plt.tight_layout()
        plt.savefig(f"{file_prefix}_super_category_{i}_weights.png")
        plt.close()


    # === Bar plots for each (super, child) category pair ===
    for j in range(struct_upbd["G2"]):  # super categories
        for i in range(struct_upbd["G1"]):  # child categories
            plt.figure(figsize=(6, 4), dpi=150)
            plt.bar(range(struct_upbd["G0"]), child_mix_weights[j][i])
            plt.title(f"Super Category {j} — Child Category {i}")
            plt.xlabel("Mixture Components")
            plt.ylabel("Weights")
            plt.ylim(0, child_mix_weights.max().item() * 1.1)
            plt.tight_layout()
            plt.savefig(f"{file_prefix}_child_category_super{j}_child{i}_weights.png")
            plt.close()


def likelihood_visualization(likelihood_mean, likelihood_std, epoch=0, log_dir="./"):
    iterations = np.arange(1, len(likelihood_mean) + 1)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
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
    save_path = os.path.join(log_dir, f"likelihood_curve_epoch_{epoch:04d}.png")
    fig.savefig(save_path, dpi=300)

    plt.close(fig)


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
