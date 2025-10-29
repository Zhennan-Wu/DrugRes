from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap


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
