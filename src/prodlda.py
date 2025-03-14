import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from wordcloud import WordCloud

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO



assert pyro.__version__.startswith('1.9.1')
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ


class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        # NB: here we set `affine=False` to reduce the number of learning parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # for the effect of this flag in BatchNorm1d
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)
    

class ProdLDAWithMLP(nn.Module):
    def __init__(self, vocab_size, interv_dim, num_topics, hidden, dropout, mlp_hidden=[100, 50]):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.interv_dim = interv_dim
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

        # Three-layer MLP for regression
        self.mlp = nn.Sequential(
            nn.Linear(num_topics + interv_dim, mlp_hidden[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden[0], mlp_hidden[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden[1], 1)  # Single output for regression
        )

    def model(self, docs, y=None):
        pyro.module("decoder", self.decoder)
        pyro.module("mlp", self.mlp)  # Register MLP in Pyro
        
        with pyro.plate("documents", docs.shape[0]):
            # Prior for topic distribution (logistic-normal)
            device = docs.device
            # print(f"doc device {device}")
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics), device=device)
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics), device=device)
            logtheta = pyro.sample("logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)  # Convert to probability simplex

            # Generate word counts (original ProdLDA)
            count_param = self.decoder(theta)
            total_count = int(docs.sum(-1).max())
            pyro.sample('obs', dist.Multinomial(total_count, count_param), obs=docs[:, 0:self.vocab_size])

            # Regression component: Predict y using MLP
            theta_r = torch.cat([theta, docs[:, self.vocab_size:]], dim=-1)
            # print(f"theta shape: {theta.shape}")
            # print(f"doc shape: {docs.shape}")
            # print(f"vocab size: {self.vocab_size}")
            # print(f"interv shape: {docs[:, self.vocab_size:].shape}")
            # print(f"theta_r shape: {theta_r.shape}")
            # print(f"Expected shape: ({docs.shape[0]}, {self.num_topics + self.interv_dim})")
            assert theta_r.shape == (docs.shape[0], self.num_topics + self.interv_dim)
            y_pred = self.mlp(theta_r).squeeze(-1)  # Forward pass through MLP
            sigma = pyro.sample("sigma", dist.HalfCauchy(1.0)).to(device)  # Uncertainty in y

            # Likelihood of y given predicted y_pred
            if y is not None:
                pyro.sample("y_obs", dist.Normal(y_pred, sigma), obs=y)

    def guide(self, docs, y=None):
        pyro.module("encoder", self.encoder)

        with pyro.plate("documents", docs.shape[0]):
            # Approximate posterior for topic proportions
            logtheta_loc, logtheta_scale = self.encoder(docs[:,0:self.vocab_size])
            pyro.sample("logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))

            # Estimate sigma for regression
            pyro.sample("sigma", dist.HalfCauchy(1.0))

    def beta(self):
        return self.decoder.beta.weight.cpu().detach().T


def plot_word_cloud(b, ax, v, n):
    sorted_, indices = torch.sort(b, descending=True)
    df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
    words = pd.merge(df, v[['index', 'word']],
                     how='left', on='index')['word'].values.tolist()
    sizes = (sorted_[:100] * 1000).int().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")


if __name__ == "__main__":
    display = True
    sample_size = 10000
    docs = torch.load("response_{}.pt".format(sample_size))
    labels_tensor = torch.load("labels_{}.pt".format(sample_size))
    vocab = pd.read_pickle("vocab_{}.pkl".format(sample_size))

    # setting global variables
    seed = 0
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_topics = 20 if not smoke_test else 3
    docs = docs.float().to(device)
    labels_tensor = labels_tensor.float().to(device)
    batch_size = 100
    learning_rate = 1e-3
    num_epochs = 500 if not smoke_test else 1

    # Clear Pyro parameter store
    pyro.clear_param_store()

    # Initialize model
    prodLDA = ProdLDAWithMLP(
        vocab_size=docs.shape[1]-2048,
        interv_dim=2048,
        num_topics=num_topics,
        hidden=100 if not smoke_test else 10,
        dropout=0.2,
        mlp_hidden=[100, 50]
    )
    prodLDA.to(device)

    # Define optimizer and loss function
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())

    # Number of batches
    num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

    # Initialize loss tracking
    elbo_losses = []
    reg_losses = []
    total_losses = []
    if (display):
        # Setup real-time plot
        plt.ion()
        
        # Create three separate figures for each loss type
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        elbo_line, = ax1.plot([], [], label="ELBO Loss", color='blue')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("ELBO Loss Curve")
        ax1.legend()
        ax1.grid()

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        reg_line, = ax2.plot([], [], label="Regression Loss", color='red')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.set_title("Regression Loss Curve")
        ax2.legend()
        ax2.grid()

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        total_line, = ax3.plot([], [], label="Total Loss", color='green')
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("Loss")
        ax3.set_title("Total Loss Curve")
        ax3.legend()
        ax3.grid()

    # Training loop
    bar = trange(num_epochs, desc="Training")
    for epoch in bar:
        running_elbo_loss = 0.0
        running_reg_loss = 0.0
        running_total_loss = 0.0

        for i in range(num_batches):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :].to(device)
            batch_labels = labels_tensor[i * batch_size:(i + 1) * batch_size].to(device)

            # Step 1: Compute ELBO Loss
            elbo_loss = svi.step(batch_docs, batch_labels) / prodLDA.vocab_size

            # Step 2: Compute Regression Loss separately
            prodLDA.eval()  # Switch to evaluation mode for prediction
            with torch.no_grad():
                theta = prodLDA.encoder(batch_docs[:, :docs.shape[1] - 2048])[0]  # Get topic distribution
                theta_r = torch.cat([theta, batch_docs[:, docs.shape[1] - 2048:]], dim=-1)
                y_pred = prodLDA.mlp(theta_r).squeeze(-1)  # Predict target values

            prodLDA.train()  # Switch back to training mode

            reg_loss = torch.nn.functional.mse_loss(y_pred, batch_labels)  # MSE Loss for regression

            # Combine losses
            total_loss = elbo_loss + reg_loss.item()
            running_elbo_loss += elbo_loss / batch_docs.size(0)
            running_reg_loss += reg_loss.item() / batch_docs.size(0)
            running_total_loss += total_loss / batch_docs.size(0)

        # Store loss values for plotting
        if (epoch > 10):
            elbo_losses.append(running_elbo_loss)
            reg_losses.append(running_reg_loss)
            total_losses.append(running_total_loss)

            bar.set_postfix(epoch_loss='{:.2e}'.format(running_total_loss))

            if (display):
                # Update plot in real-time for each figure
                elbo_line.set_xdata(np.arange(len(elbo_losses)))
                elbo_line.set_ydata(elbo_losses)
                ax1.relim()  # Adjust limits for ELBO plot
                ax1.autoscale_view()

                reg_line.set_xdata(np.arange(len(reg_losses)))
                reg_line.set_ydata(reg_losses)
                ax2.relim()  # Adjust limits for Regression plot
                ax2.autoscale_view()

                total_line.set_xdata(np.arange(len(total_losses)))
                total_line.set_ydata(total_losses)
                ax3.relim()  # Adjust limits for Total plot
                ax3.autoscale_view()

                # Draw and update each figure
                plt.draw()
                plt.pause(0.01)  # Pause to update the plot

    # Keep the final plots open
    if (display):
        plt.ioff()  # Turn off interactive mode
        plt.show()

    else:
        # Plot training loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_epochs), elbo_losses, label="ELBO Loss")
        plt.plot(range(num_epochs), reg_losses, label="Regression Loss")
        plt.plot(range(num_epochs), total_losses, label="Total Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid()
        plt.savefig("training_loss.png")  
        plt.close()    

    if not smoke_test:
        beta = prodLDA.beta()
        fig, axs = plt.subplots(7, 3, figsize=(14, 24))
        for n in range(beta.shape[0]):
            i, j = divmod(n, 3)
            plot_word_cloud(beta[n], axs[i, j], vocab, n)
        axs[-1, -1].axis('off');

        if (display):
            plt.show()
        else:
            plt.savefig("word_cloud.png")
            plt.close()