import torch
import time
from typing import Any, Union, List, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
from rbm import RBM
from utils import visualize_rbm, visualize_data, project_points_to_simplex, CSVDrugResponseDataset

from load_dataset import MNIST


class DBN:
    """
    Deep Belief Network
    """
    def __init__(self, input_size: int, layers: list, batch_size: int, learning_rate=0.1, lr_decay_factor = 0.5, lr_no_decay_length = 100, lr_decay = False, epochs: int = 10, savefile: str = None, mode: str = "bernoulli", multinomial_top: bool=False, multinomial_sample_size: int=0, bias: bool = False, k: int = 5, gaussian_top = False, top_sigma: torch.Tensor = None, sigma: torch.Tensor = None, disc_alpha: float = 1., gaussian_middle = False):
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.input_size = input_size
        self.layers = layers
        # determine
        self.bias = bias
        self.batch_size = batch_size
        self.layer_parameters = [{"W":None, "hb":None, "tW":None, "tb":None} for _ in range(len(layers))]
        self.visible_bias = None
        self.k = k
        self.mode = mode
        self.gaussian_middle = gaussian_middle
        self.gaussian_top = gaussian_top
        if (top_sigma is None):
            self.top_sigma = torch.ones((1,), dtype=torch.float32, device=self.device)/10.
        else:
            self.top_sigma = top_sigma.to(torch.float32).to(self.device)
        if (sigma is None):
            self.sigma = torch.ones((input_size,), dtype=torch.float32, device=self.device)/10.
        else:
            self.sigma = sigma.to(torch.float32).to(self.device)
        self.savefile = savefile
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.lr_no_decay_length = lr_no_decay_length
        self.lr_decay = lr_decay
        self.multinomial_top = multinomial_top
        self.multinomial_sample_size = multinomial_sample_size
        self.depthwise_training_loss = []
        self.top_parameters = {"TW":None, "Tb":None}
        self.disc_alpha = disc_alpha

    def sample_v(self, layer_index: int, y: torch.Tensor, input_mode: str="bernoulli") -> torch.Tensor:
        """
        Sample visible units given hidden units
        """
        W = self.layer_parameters[layer_index]["W"].to(torch.float32)
        if (layer_index == 0):
            vb = self.visible_bias.to(torch.float32)
            sigma = self.sigma.to(torch.float32)
        else:
            vb = self.layer_parameters[layer_index-1]["hb"].to(torch.float32)
            sigma = torch.ones((1,), dtype=torch.float32, device=self.device)/10.
            sigma = sigma.to(torch.float32)

        if (input_mode == "gaussian"):
            activation = torch.matmul(y, W)*sigma + vb
        else:
            activation = torch.matmul(y, W) + vb

        if (input_mode == "bernoulli"):
            p_v_given_h = torch.sigmoid(activation)
            variable = torch.bernoulli(p_v_given_h)
        elif (input_mode == "gaussian"):
            gaussian_dist = torch.distributions.normal.Normal(activation, sigma)
            variable = gaussian_dist.sample()
            # p_v_given_h = torch.exp(gaussian_dist.log_prob(variable))
            p_v_given_h = None
        else:
            raise ValueError("Invalid mode")
        return p_v_given_h, variable
    
    def sample_h(self, layer_index: int, x_bottom: torch.Tensor, label: torch.Tensor, input_mode: str, top_down_sample: bool=False) -> torch.Tensor:
        """
        Sample hidden units given visible units
        """
        W_bottom = self.layer_parameters[layer_index]["W"].to(torch.float32)
        bias = self.layer_parameters[layer_index]["hb"].to(torch.float32)
        if (layer_index == 0):
            if (input_mode == "gaussian"):
                activation = torch.matmul(x_bottom/self.sigma, W_bottom.t()) + bias
            else:
                activation = torch.matmul(x_bottom, W_bottom.t()) + bias
        else:    
            logistic_sigma = torch.ones((1,), dtype=torch.float32, device=self.device)/10.
            if (input_mode == "gaussian"):
                activation = torch.matmul(x_bottom/logistic_sigma, W_bottom.t()) + bias
            else:
                activation = torch.matmul(x_bottom, W_bottom.t()) + bias

        if (layer_index == len(self.layers)-1 and self.multinomial_top):
            if (top_down_sample):
                if (self.gaussian_top):
                    activation = activation + torch.matmul(label/self.top_sigma, self.top_parameters["TW"])
            p_h_given_v = torch.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device=self.device).scatter_(2, indices.unsqueeze(-1), 1)
            variable = torch.sum(one_hot, dim=1)
        else:
            p_h_given_v = torch.sigmoid(activation)
            variable = torch.bernoulli(p_h_given_v)
        return p_h_given_v, variable
    
    def sample_r(self, x_bottom: torch.Tensor) -> torch.Tensor:
        """
        Sample reconstruction
        """
        if (self.gaussian_top):
            mean = torch.mm(x_bottom, self.top_parameters["TW"].t())*self.top_sigma + self.top_parameters["Tb"]
            gaussian_dist = torch.distributions.normal.Normal(mean, self.top_sigma)
            variable = gaussian_dist.sample()
            # p_r_given_h = torch.exp(gaussian_dist.log_prob(variable))
            p_r_given_h = None
        else:
            raise ValueError("Should not sample r in this case")
            # p_r_given_h = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
            # variable = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
        return p_r_given_h, variable
        
    def generate_input_for_layer(self, index: int, dataloader: DataLoader, pretrain: bool=True) -> DataLoader:
        """
        Generate input for layer
        """
        input_layer = []
        input_labels = []
        if (index == 0):
            for batch, label in dataloader:
                input_layer.append(batch)
                input_labels.append(label)
            input_data = torch.cat(input_layer, dim=0)
            input_labels = torch.cat(input_labels, dim=0)
            if not torch.all((input_data >= 0) & (input_data <= 1)):
                raise ValueError("Tensor contains elements outside the range [0, 1].")
            dataset = TensorDataset(input_data, input_labels)
            hidden_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            return hidden_loader
        else:
            for batch, label in dataloader:
                p_x_dash, x_dash = self.generate_activation_input_for_layer(index, batch, label)
                if (pretrain):
                    input_layer.append(p_x_dash)
                else:
                    input_layer.append(x_dash)
                input_labels.append(label)
            input_data = torch.cat(input_layer, dim=0)
            input_labels = torch.cat(input_labels, dim=0)
            if not torch.all((input_data >= 0) & (input_data <= 1)):
                raise ValueError("Tensor contains elements outside the range [0, 1].")
            dataset = TensorDataset(input_data, input_labels)
            hidden_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            return hidden_loader

    def generate_activation_input_for_layer(self, index: int, dataset: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Generate input for layer
        """
        if (index == 0):
            return None, dataset
        else:
            x_gen = []
            for _ in range(self.k):
                x_dash = dataset.to(self.device)
                label = label.unsqueeze(1).to(torch.float32).to(self.device)
                for i in range(index):  
                    if (i == 0):
                        p_x, x_dash = self.sample_h(i, x_dash, label, self.mode)
                    else:
                        p_x, x_dash = self.sample_h(i, p_x, label, "gaussian")
                x_gen.append(p_x)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            x_binary = torch.bernoulli(x_dash)

            if not torch.all((x_dash >= 0) & (x_dash <= 1)):
                raise ValueError("Tensor contains elements outside the range [0, 1].")
            return x_dash.to(torch.float32).to(self.device), x_binary.to(torch.float32).to(self.device)
    
    def train(self, dataloader: DataLoader, savefig: str = None, showplot: bool = False):
        """
        Train DBN
        """
        for index, _ in enumerate(self.layers):
            start_time = time.time()
            if (index == 0):
                vn = self.input_size
            else:
                vn = self.layers[index-1]
            hn = self.layers[index]
            if (self.gaussian_middle):
                if (index == 0):
                    input_mode = self.mode
                else:
                    input_mode = "gaussian"
            else:
                input_mode = self.mode
            target_status = False
            if (index == len(self.layers)-1):
                target_status = self.gaussian_top
                if (self.multinomial_top):
                    output_mode = "multinomial"
            else:
                output_mode = "bernoulli"
            rbm_savefig = savefig + "rbm_{}/".format(index)
            rbm = RBM(n_components=hn, learning_rate=self.learning_rate, lr_decay_factor=self.lr_decay_factor, lr_no_decay_length=self.lr_no_decay_length, lr_decay=self.lr_decay, batch_size=self.batch_size, n_iter=self.epochs, verbose=0, savefile=rbm_savefig, add_bias=self.bias, target_in_model=target_status, hybrid=False, input_dist=input_mode, latent_dist=output_mode, target_dist='gaussian')

            hidden_loader = self.generate_input_for_layer(index, dataloader)

            if (index == 0):
                visible_bias = self.visible_bias
            else:
                visible_bias = self.layer_parameters[index-1]["hb"]
            rbm.fit_dataloader(hidden_loader, vn, 1, components=self.layer_parameters[index]["W"], target_components=self.layer_parameters[index]["tW"], visible_bias=visible_bias, hidden_bias=self.layer_parameters[index]["hb"], target_bias=self.layer_parameters[index]["tb"], sample_size=self.multinomial_sample_size, sigma=torch.mean(self.sigma).item(), target_sigma=torch.mean(self.top_sigma).item(), hybrid_alpha=self.disc_alpha, showplot=showplot)
            self.layer_parameters[index]["W"] = torch.tensor(rbm.components_, dtype=torch.float32, device=self.device)
            self.layer_parameters[index]["hb"] = torch.tensor(rbm.intercept_hidden_, dtype=torch.float32, device=self.device)
            self.layer_parameters[index]["tW"] = torch.tensor(rbm.target_components_, dtype=torch.float32, device=self.device)
            self.layer_parameters[index]["tb"] = torch.tensor(rbm.intercept_target_, dtype=torch.float32, device=self.device)
            if (index == 0):
                self.visible_bias = torch.tensor(rbm.intercept_visible_, dtype=torch.float32, device=self.device)
            else:
                self.layer_parameters[index-1]["hb"] = torch.tensor(rbm.intercept_visible_, dtype=torch.float32, device=self.device)
            self.top_parameters["TW"] = torch.tensor(rbm.target_components_, dtype=torch.float32, device=self.device)
            self.top_parameters["Tb"] = torch.tensor(rbm.intercept_target_, dtype=torch.float32, device=self.device)

            print("Finished Training Layer", index, "to", index+1)
            end_time = time.time()
            print("Time taken for training DBN layer", index, "to", index+1, "is", end_time-start_time, "seconds")
            visualize_rbm(rbm, hidden_loader, index, savefig, showplot)
            
        if (self.savefile is not None):
            model = self.initialize_nn_model()
            nn_savefile = self.savefile.replace(".pth", "_nn.pth")
            torch.save(model, nn_savefile)
            self.save_model()

    # def _multilayer_free_energy(self, x: torch.Tensor, y: torch.Tensor, depth: int = -1) -> torch.Tensor:
    #     """
    #     Calculate free energy
    #     """
    #     if (depth == -1):
    #         depth = len(self.layers)
    #     energy = torch.zeros(x.size(0), dtype=torch.float32, device=self.device)
    #     for i in range(depth):
    #         if (i == 0):
    #             input_mode = self.mode
    #             if (self.mode == "gaussian"):
    #                 energy = energy + torch.sum(((x_dash - self.visible_bias)/self.sigma)**2, axis=1)/2.
    #             else:
    #                 energy = energy - torch.sum(x_dash*self.visible_bias, axis=1)
    #         else:
    #             input_mode = "bernoulli"
    #         prev_x = x_dash.clone()
    #         if (i == len(self.layers)-1 and self.gaussian_top):
    #             top_down_sample = True
    #             p_x, x_dash = self.sample_h(i, x_dash, y, input_mode, top_down_sample)
    #             energy = energy + torch.sum(((x_dash - self.top_parameters["Tb"])/self.top_sigma)**2, axis=1)/2 - torch.sum(torch.matmul(x_dash, self.top_parameters["TW"].t())*y/self.top_sigma, axis=1)
    #         else:
    #             p_x, x_dash = self.sample_h(i, x_dash, y, input_mode)
    #         energy = energy - torch.sum(torch.matmul(prev_x, self.layer_parameters[i]["W"].t())*x_dash, axis=1) - torch.sum(x_dash*self.layer_parameters[i]["hb"], axis=1)
    #     return torch.mean(energy)
    
    # def estimate_multilayer_energy(self, x: torch.Tensor, y: torch.Tensor, depth: int = -1) -> torch.Tensor:
    #     """
    #     Estimate energy
    #     """
    #     if (depth == -1):
    #         depth = len(self.layers)
    #     energy = []
    #     for _ in self.k:
    #         energy.append(self._multilayer_free_energy(x, y, depth))
    #     energy = torch.stack(energy)
    #     return torch.mean(energy)
    
    # def estimate_multilayer_energy_from_loader(self, dataloader: DataLoader, depth: int = -1) -> torch.Tensor:
    #     """
    #     Estimate energy
    #     """
    #     if (depth == -1):
    #         depth = len(self.layers)
    #     energy = []
    #     for batch, label in dataloader:
    #         batch = batch.to(self.device)
    #         label = label.unsqueeze(1).to(torch.float32).to(self.device)
    #         energy.append(self.estimate_multilayer_energy(batch, label, depth))
    #     return energy

    def reconstructor(self, x: torch.Tensor, y: torch.Tensor, depth: int = -1) -> torch.Tensor:
        """
        Reconstruct input
        """
        if (depth == -1):
            depth = len(self.layers)
        x_gen = []
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(depth):
                if (i == 0):
                    input_mode = self.mode
                else:
                    input_mode = "bernoulli"
                if (i == len(self.layers)-1 and self.gaussian_top):
                    top_down_sample = True
                    p_x, x_dash = self.sample_h(i, x_dash, y, input_mode, top_down_sample)
                else:
                    p_x, x_dash = self.sample_h(i, x_dash, y, input_mode)
            x_gen.append(x_dash)
        x_dash = torch.stack(x_gen)
        x_dash = torch.mean(x_dash, dim=0)

        y_gen = []
        for _ in range(self.k):
            y_dash = x_dash
            for i in range(depth-1, -1, -1):
                p_y, y_dash = self.sample_v(i, y_dash, "bernoulli")
            y_gen.append(y_dash)
        y_dash = torch.stack(y_gen)
        y_dash = torch.mean(y_dash, dim=0)
        return y_dash, x_dash

    def reconstruct(self, dataloader: DataLoader, depth: int = -1) -> DataLoader:
        """
        Reconstruct input
        """
        visible_data = []
        latent_vars = []
        data_labels = []
        for batch, label in dataloader:
            batch = batch.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            visible, latent = self.reconstructor(batch, label, depth)
            visible_data.append(visible)
            latent_vars.append(latent)
            data_labels.append(label)
        visible_data = torch.cat(visible_data, dim=0)
        latent_vars = torch.cat(latent_vars, dim=0)
        data_labels = torch.cat(data_labels, dim=0)
        dataset = TensorDataset(visible_data, latent_vars, data_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def calc_reconstruction_error(self, dataloader: DataLoader, depth: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate reconstruction error
        """
        if (depth == -1):
            depth = len(self.layers)
        reconstruction_error = []
        for batch, label in dataloader:
            batch = batch.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            visible, _ = self.reconstructor(batch, label, depth)
            reconstruction_error.append(torch.mean((visible-batch)**2))
        return reconstruction_error
    
    def encoder(self, dataset: torch.Tensor, label: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Generate top level latent variables
        """
        dataset = dataset.to(self.device)
        activation, _ = self.generate_activation_input_for_layer(depth, dataset, label)
        if (depth == len(self.layers) and self.gaussian_top):
            activation = activation + torch.matmul(label/(self.top_sigma**2), self.top_parameters["TW"].to(self.device)) + self.top_parameters["Tb"].to(self.device)
        if (self.multinomial_top):
            p_h_given_v = torch.softmax(activation, dim=1)      
        else:
            p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v

    def encode(self, dataloader: DataLoader,  depth: int = -1) -> DataLoader:
        """
        Encode data
        """
        if (depth == -1):
            depth = len(self.layers)
        latent_vars = []
        labels = []
        for data, label in dataloader:
            data = data.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            latent_vars.append(self.encoder(data, label, depth))
            labels.append(label)
        latent_vars = torch.cat(latent_vars, dim=0)
        labels = torch.cat(labels, dim=0)
        latent_dataset = TensorDataset(latent_vars, labels)

        return DataLoader(latent_dataset, batch_size=self.batch_size, shuffle=False)
    
    def load_model(self, savefile: str, for_dbm: bool = True):
        """
        Load DBN or DBM model
        """
        model = torch.load(savefile, weights_only=False)
        if (for_dbm):
            count = 2.
        else:
            count = 1.
        layer_parameters = []
        for index in range(len(model["W"])):
            layer_parameters.append({"W":model["W"][index].to(self.device)/count, "hb":model["hb"][index].to(self.device)/count, "tW":model["tW"][index].to(self.device), "tb":model["tb"[index]].to(self.device)})
            visible_bias = model["vb"]
        
        top_parameters = {"TW":model["TW"].to(self.device), "Tb":model["Tb"].to(self.device)}
        self.layer_parameters = layer_parameters
        self.top_parameters = top_parameters
        self.visible_bias = visible_bias

    def load_nn_model(self, savefile: str, for_dbm: bool = True):
        """
        Load nn model
        """
        if (for_dbm):
            count = 2.
        else:
            count = 1.
        dbn_model = torch.load(savefile, weights_only=False)
        for layer_no, layer in enumerate(dbn_model):
            # if (layer_no//2 == len(self.layer_parameters)-1):
            #     break
            if (layer_no%2 == 0):
                self.layer_parameters[layer_no//2]["W"] = layer.weight.to(self.device)/count
                if (self.bias):
                    if (layer_no == 0):
                        self.visible_bias = layer.bias.to(self.device)/count
                    else:
                        self.layer_parameters[layer_no//2-1]["hb"] = layer.bias.to(self.device)/count
                print("Loaded Layer", layer_no//2)
        for index, layer in enumerate(self.layer_parameters):
            if (index < len(self.layer_parameters)-1):
                self.layer_parameters[index]["hb"] = self.layer_parameters[index+1]["vb"]/count

    def save_model(self, savefile: str = None):
        """
        Save model
        """
        if (savefile is None):
            savefile = self.savefile
        model = {"W": [], "vb": None, "hb": [], "tW":[], "tb":[],"TW": None , "Tb": None}
        for layer in self.layer_parameters:
            model["W"].append(layer["W"])
            model["hb"].append(layer["hb"])
            model["tW"].append(layer["tW"])
            model["tb"].append(layer["tb"])
        model["TW"] = self.top_parameters["TW"]
        model["Tb"] = self.top_parameters["Tb"]
        model["vb"] = self.visible_bias
        torch.save(model, savefile)

    def initialize_nn_model(self):
        """
        Initialize model
        """
        print("The last layer will not be activated. The rest are activated using the Sigmoid function.")

        modules = []
        for index, layer in enumerate(self.layer_parameters):
            modules.append(torch.nn.Linear(layer["W"].shape[1], layer["W"].shape[0]))
            if (index < len(self.layer_parameters)-1):
                modules.append(torch.nn.Sigmoid())
        model = torch.nn.Sequential(*modules)
        model = model.to(self.device)

        for layer_no, layer in enumerate(model):
            if (layer_no//2 == len(self.layer_parameters)-1):
                break
            if (layer_no%2 == 0):
                model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no//2]["W"])
        return model


if __name__ == "__main__":
    # mnist = MNIST()
    # train_x, train_y, test_x, test_y = mnist.load_dataset()
    # train_y = train_y/10.
    # print('MAE for all 0 selection:', torch.mean(train_x))
    data_dir = "../dataset/Cancer"
    batch_size = 100
    training_dataset = CSVDrugResponseDataset(data_dir, "training")
    data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    prev_cumu_epochs = 0
    epochs = 100
    datasize = training_dataset.length # train_x.shape[0]
    data_dimension = 5056 # train_x.shape[1]
    gaussian_middle = False
    learning_rate = 0.001
    lr_decay_factor = 0.5
    lr_no_decay_length = 100
    lr_decay = False
    
    print("The whole dataset has {} data. The dimension of each data is {}. Batch size is {}.".format(datasize, data_dimension, batch_size))
    
    # dataset = TensorDataset(train_x, train_y)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

