import torch
from hdp import HierarchicalDirichletProcess
from dbm import DBM

import torch.multiprocessing as mp

from load_dataset import MNIST

import os
import time
from typing import Any, Union, List, Tuple, Dict
from dbn_old import DBN
from rbm_old import RBM
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mp.set_start_method('spawn')
    # datasize = 1000
    # data_dimension = 784
    # latent_dimension = 10
    # dbm_stable_param = 5
    # latent_sample_size = 10
    # batch_index = 1
    # gibbs_iterations = 100

    # # Test DBM
    # dataset = torch.rand(datasize, data_dimension)
    # comparison = torch.rand(datasize, data_dimension)
    
    # dbm = DBM(data_dimension, [500, 500, latent_dimension], mode="bernoulli", k=dbm_stable_param)
    # dbm.pre_train(dataset)
    # dbm.train(dataset)
    # latent_variables = dbm.generate_top_level_latent_variables(dataset, latent_sample_size)

    # hp = HierarchicalDirichletProcess(latent_dimension, 3, datasize, 10, {2: 10})    
    # hp.gibbs_update(batch_index, gibbs_iterations, latent_variables)
    # latent_distribution = hp.get_latent_distributions()

    # reconstructed_data = dbm.generate_visible_variables(latent_distribution, latent_sample_size)

    # print("Reconstruction Difference: ", torch.mean(torch.abs(reconstructed_data.to(torch.device("cpu")) - dataset)))

    # print("Comparison Difference: ", torch.mean(torch.abs(comparison - dataset)))
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()
    print('MAE for all 0 selection:', torch.mean(train_x))

    batch_size = 1000	
    latent_dimension = 100
    hierarchical_depth = 3
    truncated_length = 10
    layer_constraints = {2: 10}
    hdp_gibbs_iterations = 51
    epoches = 10

    # train_x = train_x[:batch_size*3, :]
    # train_y = train_y[:batch_size*3]    

    datasize = train_x.shape[0]
    data_dimension = train_x.shape[1]
    print(datasize, data_dimension, batch_size)

    dataset = TensorDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dbm = DBM(data_dimension, [1000, 500, latent_dimension], batch_size, epochs = 400, savefile="dbm.pth", mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10)
    dbm.load_dbm("dbm.pth")
    latent_dataloader = dbm.encode(data_loader)


    hp = HierarchicalDirichletProcess(latent_dimension, hierarchical_depth, batch_size, truncated_length, layer_constraints)    
    hp.gibbs_dataloader_update(epoches, hdp_gibbs_iterations, latent_dataloader)
    # latent_distribution = hp.get_latent_distributions()

    # reconstructed_data = dbm.generate_visible_variables(latent_distribution, latent_sample_size)

    # print("Reconstruction Difference: ", torch.mean(torch.abs(reconstructed_data.to(torch.device("cpu")) - dataset)))

    # print("Comparison Difference: ", torch.mean(torch.abs(comparison - dataset)))