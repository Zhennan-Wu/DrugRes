import torch
import itertools
import jax
import jax.numpy as jnp

from pyro.distributions import Dirichlet, Gamma, Categorical
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Union, List, Tuple, Dict

import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
import time
import math
import random
import os
from tqdm import trange

from utils import transfer_index_tensor_to_tuple, transfer_index_tuple_to_tensor, calc_sequential_stick_breaking_weight, print_tree, INFO, HDP_DIST_INFO


PyTree = Union[jnp.ndarray, List['PyTree'], Tuple['PyTree', ...], Dict[Any, 'PyTree']]


class HierarchicalDirichletProcess:
    def __init__(self, latent_dimension: int, layers: int, batch_size: int, truncated_length: int, slot_limit: int, fixed_layers: dict = None):
        '''
        Initialize a Hierarchical Dirichlet Process with layers

        Parameters:
        - num_of_words (int): the number of words in the vocabulary
        - layers (int): the number of layers in the Hierarchical Dirichlet Process
        - fixed_layers (dict): the fixed number of categories in each layer
        - global_sample_size (int): the number of samples to draw from the Global Dirichlet Process
        '''
        #########################################################################################
        # fixed features
        ########################################################################0################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.batch_size = batch_size
        # int: the number of samples in each round
        
        self.truncate_length = truncated_length
        # int: the total number of different parameters
        
        self.latent_dimension = latent_dimension 
        # int: the dimension of each sample
        
        self.layers = layers 
        # int: the number of layers in the Hierarchical Dirichlet Process
        
        self.layer_constrains = False 
        # bool: whether there are constraints on the number of categories in each layer
        
        self.implied_constraints = None 
        # dict: the implied constraints on the number of categories in each layer
        
        self.fixed_layers = fixed_layers 
        # dict: the constrains on the number of categories in each layer

        self.gamma_alpha = 1.0

        self.gamma_beta = 1.0
        
        self._check_layer_constraints()
        # fill in the implied constraints from initial constraints input
        
        ########################################################################################
        # modifiable features from batch to batch
        ########################################################################################
        self.hyperparameters = {}
        # dict: the hyperparameters of the Hierarchical Dirichlet Process
        #  - BASE: the hyperparameter of the base dirichlet distribution
        #  - nCRP: the hyperparameter of the nested Chinese Restaurant Process
        #  - GLOBAL: the hyperparameter of the global distribution
        #  - DP: the hyperparameter of each level of Dirichlet Process

        self.number_of_subcategories = [] 
        # list of dict: the number of subcategories in each level under each category
        #  - key: the parent category
        #  - value: the number of subcategories

        self.hierarchical_observations = []
        # list of dict: the number of samples in each level under each category
        #  - key: the parent category
        #  - value: the number of samples

        self.labels_group_by_categories = []
        # list of dict: the samples belongs to each category in each level
        #  - key: the parent category
        #  - value: the samples belongs to the category

        self.hierarchical_distributions = []
        # list of dict: the hierarchical distributions of the Hierarchical Dirichlet Process
        #  - key: the parent category
        #  - value: the weights on all the available parameters (self.truncate_length is the total number of parameters)

        self.cumulative_weights = []
        # list of dict: the cumulative weights of all parameters in each category in each level across different batches
        #  - key: the parent category
        #  - value: the cumulative counts of all parameters

        self.dist_slots = []
        # list of HDP_DIST_INFO: the distribution information stored in batches to represent the model learnt distribution

        self.dist_slot_limit = slot_limit
        # int: the number of distribution information stored in the model (early batch will be substituted by the new batch if slot is full

        self.current_slot_size = 0

        self.latent_distribution_indices = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)
        # torch.Tensor: the parameter indices of each sample

        self.distributions = torch.zeros(self.batch_size, self.latent_dimension, device=self.device)
        # torch.Tensor: the parameters of each sample

        self.smallest_category_distribution_on_labels = None
        # torch.Tensor: the basic distribution over parameters of all the labels

        ################################################################################################
        # Generate all the available parameters
        ################################################################################################
        beta = Gamma(self.gamma_alpha, self.gamma_beta).sample().item()
        self.hyperparameters["BASE"] = beta
        self.parameters = self.generate_parameters()
        ################################################################################################
        # Initialize hierarchical structure
        ################################################################################################
        eta = Gamma(self.gamma_alpha, self.gamma_beta).sample().item()
        self.hyperparameters["nCRP"] = eta
        self.labels = self.generate_nCRP()
        self.labels_in_tuple = transfer_index_tensor_to_tuple(self.labels)
        self.number_of_subcategories, self.hierarchical_observations, self.labels_group_by_categories = self.summarize_group_info()
        ################################################################################################
        # Initialize the hierarchical distributions
        ################################################################################################
        beta = Gamma(self.gamma_alpha, self.gamma_beta).sample().item()
        self.hyperparameters["GLOBAL"] = beta
        self.base_weight = self.generate_base_weights()
        
        self.hyperparameters["DP"] = {}
        self.hierarchical_distributions = self.generate_hierarchical_distributions()

        self.smallest_category_distribution_on_labels, self.latent_distribution_indices, self.latent_distributions = self.summarize_distributions()

    def save_model(self, filename: str):
        '''
        Save the Hierarchical Dirichlet Process model
        '''
        model = {}
        model["hyperparameters"] = self.hyperparameters
        model["number_of_subcategories"] = self.number_of_subcategories
        model["hierarchical_observations"] = self.hierarchical_observations
        model["labels_group_by_categories"] = self.labels_group_by_categories
        model["hierarchical_distributions"] = self.hierarchical_distributions
        model["cumulative_weights"] = self.cumulative_weights
        model["latent_distribution_indices"] = self.latent_distribution_indices
        model["smallest_category_distribution_on_labels"] = self.smallest_category_distribution_on_labels
        model["labels"] = self.labels
        model["parameters"] = self.parameters
        model["latent_distributions"] = self.latent_distributions
        torch.save(model, filename)
    
    def load_model(self, filename: str):
        '''
        Load the Hierarchical Dirichlet Process model
        '''
        model = torch.load(filename)
        self.hyperparameters = model["hyperparameters"]
        self.number_of_subcategories = model["number_of_subcategories"]
        self.hierarchical_observations = model["hierarchical_observations"]
        self.labels_group_by_categories = model["labels_group_by_categories"]
        self.hierarchical_distributions = model["hierarchical_distributions"]
        self.cumulative_weights = model["cumulative_weights"]
        self.latent_distribution_indices = model["latent_distribution_indices"]
        self.smallest_category_distribution_on_labels = model["smallest_category_distribution_on_labels"]
        self.labels = model["labels"]
        self.parameters = model["parameters"]
        self.latent_distributions = model["latent_distributions"]

    def get_hyperparameters(self):
        '''
        Print the hyperparameters of the Hierarchical Dirichlet Process
        '''
        return self.hyperparameters
    
    def get_hierarchical_distributions(self):
        '''
        Print the distributions of the Hierarchical Dirichlet Process
        '''
        return self.hierarchical_distributions
    
    def get_base_weights(self):
        '''
        Print the base weights of the Hierarchical Dirichlet Process
        '''
        return self.base_weight
    
    def get_labels(self):
        '''
        Print the labels of the Hierarchical Dirichlet Process
        '''
        return self.labels
    
    def get_number_of_subcategories(self):
        '''
        Print the number of subcategories in each layer of the Hierarchical Dirichlet Process
        '''
        return self.number_of_subcategories
    
    def get_parameters(self):
        '''
        Print the parameters of the Hierarchical Dirichlet Process
        '''
        return self.parameters

    def get_latent_distributions(self):
        '''
        Print the distributions of the Hierarchical Dirichlet Process
        '''
        return self.latent_distributions
    
    def get_latent_distribution_indices(self):
        '''
        Print the distribution indices of the Hierarchical Dirichlet Process
        '''
        return self.latent_distribution_indices

    def get_smallest_category_distribution_on_labels(self):
        '''
        Print the smallest category distribution on the labels of the Hierarchical Dirichlet Process
        '''
        return self.smallest_category_distribution_on_labels

    def get_labels_group_by_categories(self):
        '''
        Print the labels grouped by categories
        '''
        return self.labels_group_by_categories
    
    def get_hierarchical_observations(self):
        '''
        Print the hierarchical observations of the Hierarchical Dirichlet Process
        '''
        return self.hierarchical_observations

    def get_cumulative_weights(self):
        '''
        Print the cumulative weights of the Hierarchical Dirichlet Process
        '''
        return self.cumulative_weights
    
    def display_update_progress(self, epoch, batch_index, round, joint_prob):
        '''
        Display the progress of the update
        '''
        # print("Update Round {}".format(round))
        # print("The number of subcategories in each layer: {}".format(self.number_of_subcategories))
        # print("The number of samples in each layer: {}".format(self.hierarchical_observations))
        # print("The labels grouped by categories: {}".format(self.labels_group_by_categories))
        
        num_iteration = [1 + round_idx for round_idx in list(range(round+1))]
        directory = '../results/plots/hdp/dataset_{}/'.format(batch_index)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        title = "Joint Probability of the Hierarchical Dirichlet Process of Epoch {} Batch {}".format(epoch, batch_index)
        if (round > 0 and round % 10 == 0):
            plt.figure()
            plt.plot(num_iteration, joint_prob)
            plt.xlabel("Number of Gibbs Sampling Iterations")
            plt.ylabel("Joint Probability")
            plt.title(title)
            plt.savefig(directory + "joint_prob_epoch{}.png".format(epoch))
            # plt.show()
            plt.close()

    def generate_parameters(self):
        '''
        Generate the parameters for the Hierarchical Dirichlet Process

        Returns:
        - parameters (torch.Tensor): the distribution parameters of the Hierarchical Dirichlet Process
        '''
        beta = self.hyperparameters["BASE"]
        base_distribution = Dirichlet(beta * torch.ones(self.latent_dimension, device = self.device))
        return base_distribution.sample((self.truncate_length, ))

    def generate_base_weights(self):
        '''
        Generate the base weights of the Hierarchical Dirichlet Process
        '''
        with torch.no_grad():
            beta = self.hyperparameters["GLOBAL"]
            remaining_weight = 1.0
            weights = []
            for _ in range(self.truncate_length):
                pi_prime = torch.distributions.Beta(1, beta).sample().item()
                pi_value = pi_prime * remaining_weight
                weights.append(pi_value)
                remaining_weight *= (1 - pi_prime)
            
            total_weight = sum(weights)
            if (total_weight > 1.0 + 1e-5):
                raise ValueError("The sum of the weights should be smaller than 1, instead got {}".format(total_weight))
            return torch.tensor(weights, device = self.device)
    
    def generate_nCRP(self):
        '''
        Generate a nested Chinese Restaurant Process with sample size sample_size and concentration parameter eta

        Returns:
        - label_hierarchy (torch.Tensor): the labels of the samples in the nested Chinese Restaurant Process
        '''
        with torch.no_grad():
            eta = self.hyperparameters["nCRP"]
            label_hierarchy = []
            seed = random.randint(0, 100)
            parent_labels = self._generate_CRP(self.batch_size, eta, seed)
            indices_group_by_categories = torch.arange(self.batch_size, device = self.device)
            parent_categories = torch.zeros(1, device = self.device)
            parent_counts = torch.tensor([self.batch_size], device = self.device)
            label_hierarchy.append(parent_labels)
            l = 1
            while (l < self.layers):
                categories, indices, counts = self._get_category_info(parent_labels, indices_group_by_categories)
                num_categories = categories.shape[0]
                if (num_categories > self.implied_constraints[l]):
                    label_hierarchy.pop()
                    if (l == 1):
                        indices = torch.arange(self.batch_size, device = self.device).unsqueeze(0)
                        categories = torch.zeros(1, device = self.device).unsqueeze(0)
                        counts = torch.tensor(self.batch_size, device = self.device).unsqueeze(0)
                        num_categories = 1
                    else:
                        indices = indices_group_by_categories
                        categories = parent_categories
                        counts = parent_counts
                        num_categories = categories.shape[0]
                    num_subcategories = self.implied_constraints[l]
                    labels =self._generate_fixed_categories(counts, eta, num_subcategories)   
                    l -= 1
                elif (l in list(self.fixed_layers.keys())):
                    num_subcategories = self.fixed_layers[l]
                    labels =self._generate_fixed_categories(counts, eta, num_subcategories)               
                else:
                    seeds = [random.randint(0, 1000000) for _ in range(num_categories)]
                    hyper_params = [eta]*num_categories
                    with Pool(num_categories) as p:
                        params = [list(item) for item in zip(counts.tolist(), hyper_params, seeds)]
                        labels = p.starmap(self._generate_CRP, params)
                if (isinstance(indices, list)):
                    global_indices = torch.cat(indices, dim=0)
                else:
                    global_indices = indices
                new_layer_label = torch.zeros(self.batch_size, dtype=torch.int, device = self.device)
                new_layer_label[global_indices] = torch.cat(labels, dim=0).int()
                label_hierarchy.append(new_layer_label)

                parent_labels = labels
                indices_group_by_categories = indices
                parent_categories = categories
                parent_counts = counts
                l += 1

            return torch.stack(label_hierarchy, dim=0).t()

    def summarize_group_info(self):
        '''
        Get the number of subcategories in the Hierarchical Dirichlet Process
        '''
        with torch.no_grad():
            number_of_subcategories = []
            hierarchical_observations = []
            labels_group_by_categories = []

            for l in range(self.layers):
                if (l != self.layers - 1):
                    child_categories = torch.unique(self.labels[:, :l+2], dim=0)
                    parent_categories, number_of_children = torch.unique(child_categories[:, :-1], dim=0, return_counts=True)
                
                    parent_keys = transfer_index_tensor_to_tuple(parent_categories)
                    one_layer_num_subcategories = dict(zip(parent_keys, number_of_children.tolist()))
                    number_of_subcategories.append(one_layer_num_subcategories)   

                parent_categories, indices, number_of_observations = torch.unique(self.labels[:, :l+1], dim=0, return_inverse = True, return_counts=True)
                parent_keys = transfer_index_tensor_to_tuple(parent_categories)

                num_observations = dict(zip(parent_keys, number_of_observations.tolist()))
                hierarchical_observations.append(num_observations)

                categorized_samples = [torch.where(indices == i)[0] for i in range(parent_categories.shape[0])]
                samples_group_by_categories = dict(zip(parent_keys, categorized_samples))
                labels_group_by_categories.append(samples_group_by_categories)

            parent_categories = transfer_index_tensor_to_tuple(torch.unique(self.labels, dim=0))
            one_layer_num_subcategories = {}
            for index_tuple in parent_categories:
                one_layer_num_subcategories[index_tuple] = 1   
            number_of_subcategories.append(one_layer_num_subcategories)

            return number_of_subcategories, hierarchical_observations, labels_group_by_categories

    def generate_hierarchical_distributions(self):
        '''
        Generate the hierarchical tree from the label hierarchy
        '''
        with torch.no_grad():
            hierarchical_distributions = []
            # First level
            child_categories = self.number_of_subcategories[0].keys()
            etas = Gamma(self.gamma_alpha, self.gamma_beta).sample((len(child_categories),)).tolist()
            hyper_params = dict(zip(child_categories, etas))
            self.hyperparameters["DP"].update(hyper_params)

            with Pool(len(child_categories)) as p:
                weights = [self.base_weight]*len(child_categories)
                truncated_lengths = [self.truncate_length]*len(child_categories)
                params = list(zip(etas, weights, truncated_lengths))
                distributions = p.starmap(calc_sequential_stick_breaking_weight, params)
            hierarchical_distributions.append(dict(zip(child_categories, distributions)))

            for l in range(self.layers-1):
                parents = self.number_of_subcategories[l].keys()
                num_childs = self.number_of_subcategories[l].values()
                children = self.number_of_subcategories[l+1].keys()
                total_num_childs = sum(num_childs)
                etas = Gamma(self.gamma_alpha, self.gamma_beta).sample((total_num_childs,)).tolist()
                hyper_params = dict(zip(children, etas))
                self.hyperparameters["DP"].update(hyper_params)

                truncated_lengths = [self.truncate_length]*total_num_childs
                parents_weights = []
                for parent, nc in zip(parents, num_childs):
                    parents_weights += [hierarchical_distributions[l][parent]]*nc
                
                with Pool(total_num_childs) as p:
                    params = list(zip(etas, parents_weights, truncated_lengths))
                    distributions = p.starmap(calc_sequential_stick_breaking_weight, params)
                hierarchical_distributions.append(dict(zip(children, distributions)))
            return hierarchical_distributions
    
    def summarize_distributions(self):
        '''
        Get the distribution of the Hierarchical Dirichlet Process
        '''
        category_distribution_on_labels = [self.hierarchical_distributions[-1][cat] for cat in self.labels_in_tuple]
        
        smallest_category_distribution_on_labels = torch.stack(category_distribution_on_labels)
        latent_distribution_indices = Categorical(smallest_category_distribution_on_labels).sample()
        latent_distributions = self.parameters[latent_distribution_indices]

        return smallest_category_distribution_on_labels, latent_distribution_indices, latent_distributions

    def posterior_update_of_params(self, concatenated_data: torch.Tensor):
        '''
        Update the parameters of the Hierarchical Dirichlet Process
        '''
        prior = self.smallest_category_distribution_on_labels.to(self.device)
        likelihood = torch.matmul(concatenated_data, self.parameters.t())
        posterior = prior * likelihood
        self.latent_distribution_indices = Categorical(posterior).sample()
        self.distributions = self.parameters[self.latent_distribution_indices]

    def update_cumulated_weights(self):
        '''
        Update the cumulated weights of the Hierarchical Dirichlet Process
        '''
        for level in range(self.layers):
            for category in self.number_of_subcategories[level].keys():
                indice = self.labels_group_by_categories[level][category]
                if (indice is None):
                    # print("The category {} in level {} is None".format(category, level))
                    # print(self.labels_group_by_categories[level])
                    continue
                else:
                    parameters = self.latent_distribution_indices[indice]
                    unique_parameters, count = torch.unique(parameters, return_counts=True)
                    param_count = torch.zeros(self.truncate_length, device = self.device)
                    param_count[unique_parameters.flatten()] += count
                    if (len(self.cumulative_weights) > level):
                        if (category not in self.cumulative_weights[level].keys()):
                            self.cumulative_weights[level][category] = param_count
                        else:
                            self.cumulative_weights[level][category] += param_count
                    else:
                        self.cumulative_weights.append({category: param_count})
                    # print("The cumulative weights of category {} in level {} is {}".format(category, level, self.cumulative_weights[level][category]))
                    # print(self.cumulative_weights)
    
    def update_batch_dist_info(self):
        '''
        Store the information of the batch
        '''
        batch_info = HDP_DIST_INFO(self.number_of_subcategories, self.labels_group_by_categories, self.latent_distribution_indices)
        if (self.current_slot_size < self.dist_slot_limit):
            self.dist_slots.append(batch_info)
            self.current_slot_size += 1
        else:
            batch_info_to_remove = self.dist_slots.pop(0)
            self.dist_slots.append(batch_info)
            self.delete_early_samples_in_cumulated_weights(batch_info_to_remove)
        
    def delete_early_samples_in_cumulated_weights(self, batch_info_to_remove: HDP_DIST_INFO):
        '''
        Denoise the cumulated weights of the Hierarchical Dirichlet Process
        '''
        for level in range(self.layers):
            for category in batch_info_to_remove.number_of_subcategories[level].keys():
                indice = batch_info_to_remove.labels_group_by_categories[level][category]
                if (indice is None):
                    continue
                else:
                    parameters = batch_info_to_remove.latent_distribution_indices[indice]
                    unique_parameters, count = torch.unique(parameters, return_counts=True)
                    param_count = torch.zeros(self.truncate_length, device = self.device)
                    param_count[unique_parameters.flatten()] += count
                    self.cumulative_weights[level][category] -= param_count

    def posterior_update_of_distributions(self):
        '''
        Update the posteriors of the Hierarchical Dirichlet Process
        '''
        with torch.no_grad():
            unique_values, counts = torch.unique(self.latent_distribution_indices, return_counts=True)
            if (len(self.cumulative_weights) == 0):
                evidence = torch.zeros(self.truncate_length, device = self.device)
            else:
                evidence = sum(self.cumulative_weights[0].values())
            evidence[unique_values] += counts
            prior_param = self.hyperparameters["GLOBAL"]
            evidence_param = torch.cat([torch.tensor(prior_param, device = self.device).unsqueeze(0), evidence], dim = 0)
            evidence_param = torch.clamp(evidence_param, min=0.1)
            evidence_weights = Dirichlet(evidence_param).sample()
            prior_weight = evidence_weights[0]
            likelihood_weight = evidence_weights[1:]
            base_weight = prior_weight * self.base_weight + likelihood_weight

            # First level
            child_categories = self.number_of_subcategories[0].keys()
            etas = [self.hyperparameters["DP"][child] + torch.sum(self._count_parameters_in_categories(child, 0)).item() for child in child_categories]
            with Pool(len(child_categories)) as p:
                weights = [base_weight]*len(child_categories)
                truncated_lengths = [self.truncate_length]*len(child_categories)
                params = list(zip(etas, weights, truncated_lengths))
                distributions = p.starmap(calc_sequential_stick_breaking_weight, params)
            self.hierarchical_distributions[0].update(dict(zip(child_categories, distributions)))
            # Other levels
            for l in range(self.layers-1):
                with Pool(len(self.number_of_subcategories[l+1].keys())) as p:
                    params = self._get_level_params_for_posterior(l+1)
                    posteriors = p.starmap(calc_sequential_stick_breaking_weight, params)
                self.hierarchical_distributions[l+1].update(dict(zip(self.number_of_subcategories[l+1].keys(), posteriors)))
            # Base distribution over parameters
            category_distribution_on_labels = [self.hierarchical_distributions[-1][cat] for cat in self.labels_in_tuple]
            self.smallest_category_distribution_on_labels = torch.stack(category_distribution_on_labels)

    def posterior_update_of_labels(self):
        '''
        Update the labels of the Hierarchical Dirichlet Process
        '''
        with torch.no_grad():
            augment_tree = self._generate_hierarchy_tree()
            v_counts, v_params, tree_labels = self._separate_trees(augment_tree)
            if (not set(self.number_of_subcategories[-1].keys()).issubset(set(tree_labels))):
                print(augment_tree)
                print(self.number_of_subcategories[-1])
                raise ValueError("The labels should be a subset of the records, instead got {}".format(set(self.number_of_subcategories[-1].keys()) - set(tree_labels)))
            tuple_labels = transfer_index_tensor_to_tuple(self.labels)
            indices = [tree_labels.index(label) for label in tuple_labels]
            v_counts[torch.arange(self.batch_size, device = self.device), torch.tensor(indices, device = self.device)] -= 1
            likelihood = v_params[torch.arange(self.batch_size, device = self.device), self.latent_distribution_indices, :]
            posterior = torch.clamp(likelihood * v_counts, min=1e-3)
            new_label_indices = Categorical(posterior/posterior.sum(dim=1, keepdim=True)).sample()
            new_labels = [tree_labels[idx] for idx in new_label_indices.tolist()]
            label_ref = set(tuple_labels)
            new_label_ref = set(new_labels)
            if (not new_label_ref.issubset(label_ref)):
                # Generate new dsitributions and corresponding counts
                new_categories = list(new_label_ref - label_ref)
                self._increase_categories(new_categories, new_labels)

            self.labels = torch.tensor([[int(index) for index in label] for label in new_labels], device = self.device)

    def gibbs_update(self, epoch: int, batch_index: int, number_of_iterations: int, data: torch.Tensor, test: bool):
        '''
        Update the Hierarchical Dirichlet Process using Gibbs Sampling
        '''
        joint_prob = []
        data = data.to(self.device)
        concatenated_data = torch.sum(data, dim=1)
        if (not test):
            learning = trange(number_of_iterations, desc=str("Gibbs Update of Epoch {} Batch {} Starting...".format(epoch, batch_index)))
        else:
            learning = trange(number_of_iterations, desc=str("Gibbs Update of Test Data Batch {} Starting...".format(batch_index)))
        for round in learning:
            self.posterior_update_of_params(concatenated_data)
            self.posterior_update_of_distributions()
            self.posterior_update_of_labels()
            joint_prob.append(self.calculate_joint_probability(concatenated_data))
            self.display_update_progress(epoch, batch_index, round, joint_prob)
            learning.refresh()           
        learning.close()
        if (not test):
            self.update_cumulated_weights()
            self.update_batch_dist_info()

    def infer_dataloader(self, number_of_iterations: int, dataloader: DataLoader):
        '''
        Infer the Hierarchical Dirichlet Process using Gibbs Sampling with a dataloader
        '''
        batch_index = 0
        directory = '../results/printouts/hdp/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for data, label in dataloader:
            batch_index += 1
            self.gibbs_update(0, batch_index, number_of_iterations, data, True)
            file = directory + "hdp_batch{}.txt".format(batch_index)
            self.display_hierarchical_results(file, label)

    def gibbs_dataloader_update(self, epochs: int, number_of_iterations: int, dataloader: DataLoader, test: bool = False):
        '''
        Update the Hierarchical Dirichlet Process using Gibbs Sampling with a dataloader
        '''
        for epoch in range(epochs):
            batch_index = 0
            for data, _ in dataloader:
                batch_index += 1
                self.gibbs_update(epoch, batch_index, number_of_iterations, data, test)
        if (not test):
            self.save_model("hdp.pth")

    def calculate_joint_probability(self, concatenated_data: torch.Tensor):
        '''
        Calculate the joint probability of the Hierarchical Dirichlet Process
        '''
        with torch.no_grad():
            marginalize_out_topic = torch.matmul(self.smallest_category_distribution_on_labels.to(self.device), self.parameters.to(self.device))
            joint_prob = concatenated_data * marginalize_out_topic
            return torch.sum(joint_prob).item()

    def summarize_hierarchical_results(self, ground_truth: torch.Tensor) -> dict:
        '''
        Generate the results of the Hierarchical Dirichlet Process
        '''
        root = {}
        tree_level = root

        for l in range(self.layers-1):
            if (l == 0):
                for cc in self.number_of_subcategories[l+1].keys():
                    pc = cc[:-1]
                    if (pc not in tree_level.keys()):
                        tree_level[pc] = {cc: {}}
                    else:
                        tree_level[pc][cc] = {}
                tree_level = tree_level.values()
            else:
                new_level = []
                for tree in tree_level:
                    for cc in self.number_of_subcategories[l+1].keys():
                        pc = cc[:-1]
                        if (pc in tree.keys()):
                            tree[pc][cc] = {}
                    new_level += list(tree.values())
                tree_level = new_level

        for tree in tree_level:
            total_leaf_keys = list(self.number_of_subcategories[-1].keys())
            for cc in total_leaf_keys:
                if (cc in tree.keys()):
                    if (ground_truth is not None):
                        tree[cc] = {"labels": self.labels_group_by_categories[-1][cc], "weights": self.hierarchical_distributions[-1][cc], "true_labels": ground_truth[self.labels_group_by_categories[-1][cc]]}
                    else:
                        tree[cc] = {"labels": self.labels_group_by_categories[-1][cc], "weights": self.hierarchical_distributions[-1][cc]}
        return root
    
    def display_hierarchical_results(self, file: str, ground_truth: torch.Tensor = None):
        '''
        Display the hierarchical results of the Hierarchical Dirichlet Process
        '''
        label_hierarchy = self.summarize_hierarchical_results(ground_truth)
        print_tree(label_hierarchy, file)

    def _increase_categories(self, new_categories: list, new_labels: list): 
        '''
        Increase the categories of the Hierarchical Dirichlet Process
        '''
        for new_cat in new_categories:
            for pi in range(len(new_cat)):
                parent_cat = new_cat[:pi+1]
                if (parent_cat not in self.hierarchical_distributions[pi].keys()):
                    if (pi > 0):
                        self.hierarchical_distributions[pi][parent_cat] = calc_sequential_stick_breaking_weight(self.hyperparameters["DP"][parent_cat], self.hierarchical_distributions[pi-1][parent_cat[:-1]], self.truncate_length)
                    else:
                        self.hierarchical_distributions[pi][parent_cat] = calc_sequential_stick_breaking_weight(self.hyperparameters["DP"][parent_cat], self.base_weight, self.truncate_length)
                if (parent_cat not in self.number_of_subcategories[pi].keys()):
                    self.number_of_subcategories[pi][parent_cat] = 1
                    if (pi > 0):
                        self.number_of_subcategories[pi-1][parent_cat[:-1]] += 1

        label_hierarchy = transfer_index_tuple_to_tensor(new_labels)
        for l in range(self.layers):
            parent_categories, indices, number_of_observations = torch.unique(label_hierarchy[:, :l+1], dim=0, return_inverse = True, return_counts=True)
            parent_keys = transfer_index_tensor_to_tuple(parent_categories)
            disappeared_categories = list(set(self.hierarchical_observations[l].keys()) - set(parent_keys))
            disappeared_counts = torch.zeros(len(disappeared_categories), device = self.device)

            parent_keys += disappeared_categories
            number_of_observations = torch.cat([number_of_observations.to(self.device), disappeared_counts])
            num_observations = dict(zip(parent_keys, number_of_observations.tolist()))
            self.hierarchical_observations[l].update(num_observations)

            categorized_samples = [torch.where(indices == i)[0] for i in range(parent_categories.shape[0])]
            categorized_samples += [None]*len(disappeared_categories)
            samples_group_by_categories = dict(zip(parent_keys, categorized_samples))
            self.labels_group_by_categories[l].update(samples_group_by_categories)

    def _generate_hierarchy_tree(self, debug: bool = False):
        '''
        Generate the distribution tree from the hierarchical distributions
        '''
        root = {}
        tree_level = root

        for l in range(self.layers-1):
            if (l == 0):
                for cc in self.number_of_subcategories[l+1].keys():
                    pc = cc[:-1]
                    if (pc not in tree_level.keys()):
                        tree_level[pc] = {cc: {}}
                    else:
                        tree_level[pc][cc] = {}
                if (len(self.cumulative_weights) > 0):
                    for cc in self.cumulative_weights[l+1].keys():
                        pc = cc[:-1]
                        if (pc not in tree_level.keys()):
                            tree_level[pc] = {cc: {}}
                        else:
                            tree_level[pc][cc] = {}
                num_categories = len(tree_level.keys()) 
                if (num_categories < self.implied_constraints[l]):
                    tree_level[(num_categories,)] = {'parent': (num_categories,)}
                    self.hyperparameters["DP"][(num_categories,)] = Gamma(self.gamma_alpha, self.gamma_beta).sample().item()
                tree_level = tree_level.values()
                if (debug):
                    print(tree_level)
            else:
                new_level = []
                total_num_categories = sum([len(tree.keys()) for tree in tree_level])
                for tree in tree_level:
                    for cc in self.number_of_subcategories[l+1].keys():
                        pc = cc[:-1]
                        if (pc in tree.keys()):
                            tree[pc][cc] = {}
                    if (len(self.cumulative_weights) > 0):
                        for cc in self.cumulative_weights[l+1].keys():
                            pc = cc[:-1]
                            if (pc in tree.keys()):
                                tree[pc][cc] = {}
                    if ('parent' in tree.keys()):
                        prefix = tree.pop('parent')
                        num_categories =  (0,)
                        total_num_categories -= 1
                    else:
                        prefix = list(tree.keys())[0][:-1]
                        num_categories = (self.number_of_subcategories[l-1][prefix],)
                    if (total_num_categories < self.implied_constraints[l]):
                        new_value = prefix + num_categories
                        tree[new_value] = {'parent': new_value}
                        self.hyperparameters["DP"][new_value] = Gamma(self.gamma_alpha, self.gamma_beta).sample().item()
                    new_level += list(tree.values())
                tree_level = new_level
                if (debug):
                    print(tree_level)
        total_num_categories = sum([len(tree.keys()) for tree in tree_level])
        for tree in tree_level:
            if (len(self.cumulative_weights) > 0):
                total_leaf_keys = set(list(self.number_of_subcategories[-1].keys()) + list(self.cumulative_weights[-1].keys()))
            else:
                total_leaf_keys = list(self.number_of_subcategories[-1].keys())
            for cc in total_leaf_keys:
                if (cc in tree.keys()):
                    observation = torch.tensor([0.]) 
                    if (len(self.cumulative_weights) > 0):
                        if (cc in self.cumulative_weights[-1].keys()):
                            observation += torch.sum(self.cumulative_weights[-1][cc])
                    if (cc in self.hierarchical_observations[-1].keys()):
                        observation += self.hierarchical_observations[-1][cc]
                    tree[cc] = INFO(observation, cc, self.hierarchical_distributions[-1][cc].to(self.device))
            if ('parent' in tree.keys()):
                prefix = tree.pop('parent')
                num_categories = (0,)
                total_num_categories -= 1
            else:
                prefix = list(tree.keys())[0][:-1]
                num_categories = (self.number_of_subcategories[-2][prefix],)
            if (total_num_categories < self.implied_constraints[self.layers - 1]):
                new_value = prefix + num_categories
                tree[new_value] = INFO(self.hyperparameters["nCRP"], new_value, self.base_weight.to(self.device))
                self.hyperparameters["DP"][new_value] = Gamma(self.gamma_alpha, self.gamma_beta).sample().item()
        if (debug):
            print(tree_level)
        return root

    def _separate_trees(self, augment_tree):
        '''
        Separate the tree into subtrees
        '''
        counts = []
        params = []
        labels = []
        flatten_leaves = jax.tree.leaves(augment_tree)
        for leaf in flatten_leaves:
            counts.append(leaf.get_count())
            labels.append(leaf.get_label())
            params.append(leaf.get_param())
        counts = torch.tensor(counts, device = self.device)
        params = torch.stack(params).t()
        vectorized_counts = torch.stack([counts]*self.batch_size)
        vectorized_params = torch.stack([params]*self.batch_size)
        return vectorized_counts, vectorized_params, labels

    def _check_layer_constraints(self):
        '''
        Check if the layer constraints are satisfied
        '''
        if (self.fixed_layers is not None):
            fix_keys = list(self.fixed_layers.keys())
            fix_values = list(self.fixed_layers.values())
            if (len(fix_keys) > 1):
                if (all(fix_keys[i] <= fix_keys[i+1] for i in range(len(fix_keys)-1))):
                    raise ValueError("The fixed layers should be in increasing order, get {}".format(fix_keys))
                if (all(fix_values[i] <= fix_values[i+1] for i in range(len(fix_values)-1))):
                    raise ValueError("The fixed layers should have increasing number of categories, get {}".format(fix_values))
            self.layer_constrains = True
            layer_index = [float('inf')]*self.layers
            self.implied_constraints = {}
            for i in range(self.layers):
                if (i in self.fixed_layers.keys()):
                    layer_index[i] = self.fixed_layers[i]
            for i in range(self.layers):
                self.implied_constraints[i] = min(layer_index[i:])

    def _get_level_params_for_posterior(self, level: int):
        '''
        Get the parameters for the posterior of the Hierarchical Dirichlet Process

        Parameters:
        - level (int): the level of the Hierarchical Dirichlet Process

        Returns:
        - prior_param (torch.Tensor): the prior parameters of the Hierarchical Dirichlet Process
        - evidence_param (torch.Tensor): the evidence parameters of the Hierarchical Dirichlet Process
        '''
        parent_categories = self.number_of_subcategories[level-1].keys()
        child_categories = self.number_of_subcategories[level].keys()
        parent_child_pairs = {}
        for pc in parent_categories:
            parent_child_pairs[pc] = []
            for cc in child_categories:
                if (pc == cc[:-1]):
                    parent_child_pairs[pc].append(cc)
        params = []
        for child in child_categories:
            count = self._count_parameters_in_categories(child, level).to(torch.device("cpu"))
            prior_param = self.hyperparameters["DP"][child]
            parent_dist = self.hierarchical_distributions[level-1][child[:-1]].to(torch.device("cpu"))
            params.append(tuple([count.sum().item()+prior_param, (parent_dist * prior_param + count)/(prior_param + count.sum().item()), self.truncate_length]))
        return params

    def _count_parameters_in_categories(self, category: str, level: int):
        '''
        Count the number of parameters in the categories

        Parameters:
        - categories (str): the categories to count the parameters
        - level (int): the level of the Hierarchical Dirichlet Process

        Returns:
        - num_parameters (int): the number of parameters in the categories
        '''
        indice = self.labels_group_by_categories[level][category]
        if (indice is None):
            if (len(self.cumulative_weights) == 0):
                param_count = torch.zeros(self.truncate_length, device = self.device)
            else:   
                if (category not in self.cumulative_weights[level].keys()):
                    param_count = torch.zeros(self.truncate_length, device = self.device)
                else:
                    param_count = self.cumulative_weights[level][category]        
        else:
            parameters = self.latent_distribution_indices[indice]
            unique_parameters, count = torch.unique(parameters, return_counts=True)
            if (len(self.cumulative_weights) == 0):
                param_count = torch.zeros(self.truncate_length, device = self.device)
            else:   
                if (category not in self.cumulative_weights[level].keys()):
                    param_count = torch.zeros(self.truncate_length, device = self.device)
                else:
                    param_count = self.cumulative_weights[level][category]
            param_count[unique_parameters.flatten()] += count
        return param_count
        
    def _get_category_info(self, labels: Union[torch.Tensor, list], indices: Union[torch.Tensor, list]):
        '''
        Summarize the Chinese Restaurant Process to get the unique values and their counts

        Parameters:
        - labels (torch.Tensor or list): the labels of the samples
        - indices (torch.Tensor or list): the indices of the samples that belongs to each parent category

        Returns:
        - categories (torch.Tensor): all different categories among different parent categories
        - samples_group_by_category (list): a list contains the indices of the samples that belongs to each parent category
        - counts_by_category (torch.Tensor): the counts of the samples that belongs to each category
        '''
        category_list = []
        samples_group_by_category = []
        counts_by_category_list = []
        if (isinstance(labels, list)):
            for label, index in zip(labels, indices):
                categories, sample_category_assignments, counts_by_category = torch.unique(label, return_inverse = True, return_counts=True)
                for i in range(categories.shape[0]):
                    samples_group_by_category.append(index[torch.where(sample_category_assignments == i)])
                category_list.append(categories)
                counts_by_category_list.append(counts_by_category)
            categories = torch.cat(category_list, dim=0)
            counts_by_category = torch.cat(counts_by_category_list, dim=0)
        else:
            categories, sample_category_assignments, counts_by_category = torch.unique(labels, return_inverse = True, return_counts=True)
            for i in range(categories.shape[0]):
                samples_group_by_category.append(indices[torch.where(sample_category_assignments == i)])
        return categories, samples_group_by_category, counts_by_category
    
    def _generate_CRP(self, sample_size: int, eta: float, random_seed: int):
        '''
        Generate a Chinese Restaurant Process with sample size sample_size and concentration parameter eta
        
        Parameters:
        - sample_size (int): the number of samples to generate labels for
        - eta (float): the concentration parameter of the Chinese Restaurant Process

        Returns:
        - labels (torch.Tensor): the labels of the samples
        '''
        labels = torch.tensor([0], dtype=torch.int32, device = self.device)
        for _ in range(1, sample_size):
            categories, counts = torch.unique(labels, return_counts=True)
            if (random.random() < eta/(eta + torch.sum(counts).item())):
                # Add a new label
                new_label = torch.max(categories) + 1
                labels = torch.cat((labels, new_label.unsqueeze(0)), dim=0)
            else:
                # Select an existing label
                index = Categorical(counts).sample()
                new_label = categories[index.int()]
                labels = torch.cat((labels, new_label.unsqueeze(0)), dim=0)
        return labels

    def _generate_fixed_categories(self, parent_categories_counts: torch.Tensor, eta: float, num_categories: int):
        '''
        Generate a Chinese Restaurant Process with sample size sample_size and concentration parameter eta with fixed number of categories

        Parameters:
        - parent_categories_counts (torch.Tensor): the number of samples in each parent category
        - eta (float): the concentration parameter of the Chinese Restaurant Process
        - num_categories (int): the number of child categories to generate

        Returns:
        - labels (torch.Tensor): the labels of the samples
        '''
        num_parent_categories = parent_categories_counts.shape[0]
        if (num_categories < num_parent_categories):
            raise ValueError("The number of child categories should be greater than the number of parent categories")
        # Randomly assign child categories to parent categories
        child_categories = list(range(num_categories))
        child_categories = [[x] for x in child_categories]
        parent_child_relation = dict(zip(list(range(num_categories)),child_categories))
        additional_categories = num_categories - num_parent_categories
        for i in range(additional_categories):
            parent_category = torch.randint(0, num_parent_categories, (1,), device = self.device)
            parent_child_relation[parent_category.item()].append(i+num_parent_categories)
        parent_categories_counts = parent_categories_counts.to(torch.int).tolist()
        parent_child_list = list(parent_child_relation.values())

        # Generate instances based on the assigned parent-child categories
        labels = []
        for p_count, p_c in zip(parent_categories_counts, parent_child_list):
            # Generate the first instance under the parent category
            p_labels = []
            p_labels_record = []
            candidates = p_c
            new_label = candidates[torch.randint(0, len(candidates), (1,)).item()]
            p_labels_record.append(new_label) 
            p_labels.append(0)    
            # Chinese resutaurant process to generate the rest of the instances      
            for _ in range(p_count-1):
                counts = len(set(p_labels_record))
                candidates = list(set(p_c) - set(p_labels_record))
                if (random.random() < eta/(eta + counts) and len(candidates) > 0):
                    # Add a new label
                    new_label = candidates[random.randint(0, len(candidates)-1)]
                    p_labels_record.append(new_label)
                    p_labels.append(max(p_labels)+1)
                else:
                    categories, counts = torch.unique(torch.tensor(p_labels, device = self.device), return_counts=True)
                    new_label_index = Categorical(counts).sample()
                    new_label = categories[new_label_index].item()
                    p_labels.append(new_label)
            labels.append(torch.tensor(p_labels, device = self.device))
        return labels
    

def generate_pseudo_samples(data_sizes: list, dimension: int, sample_size: int):
    '''
    '''
    categories = len(data_sizes)
    variable_size = int(dimension/categories)
    data = []
    labels = []
    ordered_categories = list(range(categories))
    # random.shuffle(ordered_categories)
    # print(ordered_categories)
    for cat_size, index in zip(data_sizes, ordered_categories):
        for _ in range(cat_size):
            new_data = torch.zeros((sample_size, dimension), dtype = torch.int)
            indices = torch.randint(0, variable_size, (sample_size,))
            one_hot_vectors = torch.zeros(sample_size, variable_size)
            one_hot_vectors.scatter_(1, indices.unsqueeze(1), 1)
            new_data[:, index*variable_size:(index+1)*variable_size] = one_hot_vectors
            data.append(new_data)
            labels.append(index)
    data = torch.stack(data, dim=0).to(torch.float)
    labels = torch.tensor(labels, dtype = torch.float)
    return data, labels

        
if __name__ == "__main__":

    input_dimen = 10
    batch_size = 50
    number_of_latent_sample = 10
    data_sizes = [1050, 100, 1050]
    slot_limit = min(int(sum(data_sizes)/batch_size), 50)

    # random.shuffle(data_sizes)
    # print(data_sizes)
    train_x, train_y = generate_pseudo_samples(data_sizes, input_dimen, number_of_latent_sample)
    gibbs_sampling_iterations = 100

    latentloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    hp = HierarchicalDirichletProcess(latent_dimension=input_dimen, layers=3, batch_size=batch_size, truncated_length=10, slot_limit=slot_limit, fixed_layers={2: 9})
    hp.gibbs_dataloader_update(epochs=100, number_of_iterations=gibbs_sampling_iterations, dataloader=latentloader)
    hp.infer_dataloader(number_of_iterations=gibbs_sampling_iterations, dataloader=latentloader)

