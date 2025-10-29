import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import binarize
from sklearn.utils import shuffle
from umap import UMAP


def visualize_data(X_train_embedded, y_train, rbm_type, savefig="../plts/", showplot=True):
    
    umap = UMAP()
    # Fit and transform the data
    X_train_umap = umap.fit_transform(X_train_embedded)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap="Spectral", s=1, alpha=0.6)
    plt.colorbar(scatter, label="Digit Label")
    plt.title(f"UMAP Encoding of Layer {rbm_type[-1]}: {rbm_type[:-2].upper()}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    if (showplot):
        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(savefig+rbm_type+".png")
    plt.close()


def fit_dataloader(dataloader, model_pipeline, epochs, savefile="dbn_", showplot=True):
    """
    Fit each RBM layer in a DBN pipeline using batch-wise training from a PyTorch DataLoader.
    """
    energy_tracking_per_layer = []

    # Access RBM layers from the pipeline
    rbm_layers = [step[1] for step in model_pipeline.steps]
    rbm_types =  [step[0] for step in model_pipeline.steps]

    input_data = []
    input_labels = []

    # Collect all batches into a single numpy array (required for layer-wise pretraining)
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            X = batch[0]
            y = batch[1].reshape(-1,)
        else:
            X = batch
            y = np.zeros(len(X), )
        input_data.append(X.detach().cpu().numpy())
        input_labels.append(y)
    input_data = np.vstack(input_data)
    input_labels = np.hstack(input_labels)

    # Train each layer one by one
    for layer_idx, rbm in enumerate(rbm_layers):
        print(f"\nTraining RBM Layer {layer_idx+1} {rbm_types[layer_idx]} with {rbm.n_components} hidden units")
        if not "gaussian" in rbm_types[layer_idx]:
            input_data = binarize(input_data, input_data.mean())
            "Input data binarized"
        energy_mean_tracking = []
        energy_var_tracking = []

        for epoch in range(epochs):
            energy_epoch = []
            # Shuffle the full data for each epoch

            input_data, input_labels = shuffle(input_data, input_labels, random_state=layer_idx+epoch)


            for i in range(0, len(input_data), rbm.batch_size):
                batch = input_data[i:i + rbm.batch_size]
                if batch.shape[0] < rbm.batch_size:
                    continue  # Skip incomplete batch

                rbm.partial_fit(batch)

                if hasattr(rbm, 'score_samples'):
                    energy = -rbm.score_samples(batch)
                    energy_epoch.extend(energy)

            energy_epoch = np.array(energy_epoch)
            energy_mean_tracking.append(energy_epoch.mean())
            energy_var_tracking.append(energy_epoch.var())

        energy_tracking_per_layer.append((energy_mean_tracking, energy_var_tracking))

        # Transform data for next layer
        input_data = rbm.transform(input_data)

        visualize_data(input_data, input_labels, rbm_types[layer_idx], showplot)


    # Plot energy per layer
    for i, (means, vars_) in enumerate(energy_tracking_per_layer):
        x = np.arange(1, len(means) + 1)
        y = np.array(means)
        y_var = np.array(vars_)
        y_upper = y + y_var
        y_lower = y - y_var

        plt.plot(x, y, label=f"Layer {i+1} Mean Energy")
        plt.fill_between(x, y_lower, y_upper, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title("Energy vs Epoch per Layer")
    plt.legend()

    if showplot:
        plt.show()
    else:
        plt.savefig(f"{savefile}energy.png")
    plt.close()
