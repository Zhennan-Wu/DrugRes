import torch
import numpy as np
import os

from dbn import DBN
from load_mut_drug import load_data
from utils import visualize_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


batch_size = 100
datasize = 100000
data_loader = load_data(mut_len=3008, data_size=datasize, batch_size=batch_size)


prev_cumu_epochs = 0
epochs = 500
data_dimension = 5056 
gaussian_middle = False
learning_rate = 0.001
lr_decay_factor = 0.5
lr_no_decay_length = 100
lr_decay = False

print("The whole dataset has {} data. The dimension of each data is {}. Batch size is {}.".format(datasize, data_dimension, batch_size))

for experiment in ["bernoulli", "multinomial"]:
    directory = "../results/plots/DBN_ccl/epoch_{}/".format(epochs + prev_cumu_epochs)
    experi_type = experiment
    directory = directory + "UMAP_ccl_" + experi_type + "/"
    filename = "dbn_ccl_" + experi_type + ".pth"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if (experiment == "bernoulli"):
        dbn = DBN(data_dimension, layers=[2000, 500, 100], batch_size=batch_size, learning_rate=learning_rate, lr_decay_factor=lr_decay_factor, lr_no_decay_length=lr_no_decay_length, lr_decay=lr_decay, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = False, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = False, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1., gaussian_middle = gaussian_middle)
    elif (experiment == "bernoulli_label"):
        dbn = DBN(data_dimension, layers=[500, 300, 100], batch_size=batch_size, learning_rate=learning_rate, lr_decay_factor=lr_decay_factor, lr_no_decay_length=lr_no_decay_length, lr_decay=lr_decay, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = False, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = True, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1., gaussian_middle = gaussian_middle)
    elif (experiment == "multinomial"):
        dbn = DBN(data_dimension, layers=[2000, 500, 100], batch_size=batch_size, learning_rate=learning_rate, lr_decay_factor=lr_decay_factor, lr_no_decay_length=lr_no_decay_length, lr_decay =lr_decay, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = False, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1., gaussian_middle = gaussian_middle)
    elif (experiment == "multinomial_label"):
        dbn = DBN(data_dimension, layers=[500, 300, 100], batch_size=batch_size, learning_rate=learning_rate, lr_decay_factor=lr_decay_factor, lr_no_decay_length=lr_no_decay_length, lr_decay=lr_decay, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = True, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1., gaussian_middle = gaussian_middle)
    else:
        raise ValueError("Invalid Experiment Type")
    # dbn.load_model(filename)
    dbn.train(data_loader, directory, showplot=False)

    latent_loader = dbn.encode(data_loader)
    new_dir = directory + "final_latent_embedding.png"
    visualize_data(latent_loader, 3, new_dir)
    print("Finished {} Experiment".format(experiment))