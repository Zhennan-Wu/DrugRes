import torch
import pyro
import random
from umap import UMAP
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd


from pyro.distributions import Categorical
from torch.utils.data import Dataset
from typing import Any, Union, List, Tuple, Dict
   

class HDP_DIST_INFO:
    def __init__(self, number_of_subcategories: dict, labels_group_by_categories: dict, latent_distribution_indices: torch.Tensor) -> None:
        self.number_of_subcategories = number_of_subcategories
        self.labels_group_by_categories = labels_group_by_categories
        self.latent_distribution_indices = latent_distribution_indices
    
    def get_number_of_subcategories(self):
        return self.number_of_subcategories
    
    def get_labels_group_by_categories(self):
        return self.labels_group_by_categories
    
    def get_latent_distribution_indices(self):
        return self.latent_distribution_indices
    
    
class INFO:
    def __init__(self, count: int, label: torch.Tensor, param: torch.Tensor) -> None:
        self.count = count
        self.label = label
        self.param = param
    
    def get_count(self):
        return self.count
    
    def get_label(self):
        return self.label
    
    def get_param(self):
        return self.param
    

def calc_sequential_stick_breaking_weight(alpha: float, parent_weights: list, num_categories: int):
    '''
    Generate the hierarchical distributions
    '''
    child_weights = []
    v_values = []
    concentrate1 = alpha
    for k in range(num_categories):
        concentrate0 = alpha*parent_weights[k]
        concentrate1 -= concentrate0
        concentrate0 = max(concentrate0, 1e-3)
        concentrate1 = max(concentrate1, 1e-3)
        v_prime = torch.distributions.Beta(concentrate0, concentrate1).sample().item()
        v_values.append(v_prime)
        pi_final = v_prime
        for j in range(k):
            pi_final *= 1. - v_values[j]
        child_weights.append(pi_final)
    return torch.tensor(child_weights)


def transfer_index_tensor_to_tuple(index: torch.Tensor):
    '''
    Transfer the index tensor to string
    '''
    index_list = index.tolist()
    index_tuple = []
    for index in index_list:
        if (isinstance(index, int)):
            index_tuple.append((index,))
        else:
            index_tuple.append(tuple(index))
    return index_tuple


def transfer_index_tuple_to_tensor(indices: list):
    '''
    Transfer the index tuple to tensor
    '''
    index_tensor = []
    for index in indices:
        index_tensor.append(torch.tensor(index))
    return torch.stack(index_tensor)


def print_tree(d, file:str=None, indent:int =0, opened:bool = False):
    # Loop through dictionary items
    if (file != None):
        if (not opened):
            opened = True
            with open(file, 'w') as f:
                print("Reconstructed HDP distribution tree:", file=f)
            for key, value in d.items():
                # Print the key with proper indentation
                print('    ' * indent + str(key), file=open(file, 'a'))
                
                # If value is another dictionary, recursively call the function
                if isinstance(value, dict):
                    print_tree(value, file, indent + 1, opened)
                # If the value is a tensor, print its content
                elif isinstance(value, torch.Tensor):
                    print('    ' * (indent + 1) + f'Tensor: {value}', file=open(file, 'a'))
                else:
                    # Print the value with additional indentation
                    print('    ' * (indent + 1) + str(value), file=open(file, 'a'))
        else:
            for key, value in d.items():
                # Print the key with proper indentation
                print('    ' * indent + str(key), file=open(file, 'a'))
                # If value is another dictionary, recursively call the function
                if isinstance(value, dict):
                    print_tree(value, file, indent + 1, opened)
                # If the value is a tensor, print its content
                elif isinstance(value, torch.Tensor):
                    print('    ' * (indent + 1) + f'Tensor: {value}', file=open(file, 'a'))
                else:
                    # Print the value with additional indentation
                    print('    ' * (indent + 1) + str(value), file=open(file, 'a'))            
    else:  
        for key, value in d.items():
            # Print the key with proper indentation
            print('    ' * indent + str(key))
            
            # If value is another dictionary, recursively call the function
            if isinstance(value, dict):
                print_tree(value, indent + 1)
            # If the value is a tensor, print its content
            elif isinstance(value, torch.Tensor):
                print('    ' * (indent + 1) + f'Tensor: {value}')
            else:
                # Print the value with additional indentation
                print('    ' * (indent + 1) + str(value))


def visualize_rbm(rbm, hidden_loader, level, savefig, showplot=False):
    X_train_bin, y_train = hidden_loader.dataset.tensors
    X_train_bin = X_train_bin.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    y_train_input = y_train.reshape(X_train_bin.shape[0], -1) / 10.0 
    X_train_embedded = rbm.transform(X_train_bin, y_train_input)
    umap = UMAP()
    # Fit and transform the data
    X_train_umap = umap.fit_transform(X_train_embedded)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap="Spectral", s=1, alpha=0.6)
    plt.colorbar(scatter, label="Digit Label")
    plt.title("UMAP RBM Embedding of MNIST Training Data of level {}".format(level))
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    # plt.show()
    filename = savefig + "level_" + str(level) + ".png"
    if (showplot):
        # Add a caption
        plt.figtext(0.5, 0.02, filename, ha='center', fontsize=10, color='gray')

        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


def visualize_data(hidden_loader, level, savefig, showplot=False):
    X_train_bin, y_train = hidden_loader.dataset.tensors
    X_train_embedded = X_train_bin.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    y_train_input = y_train.reshape(X_train_bin.shape[0], -1) / 10.0
    
    umap = UMAP()
    # Fit and transform the data
    X_train_umap = umap.fit_transform(X_train_embedded)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap="Spectral", s=1, alpha=0.6)
    plt.colorbar(scatter, label="Digit Label")
    plt.title("UMAP RBM Encoding of MNIST Training Data of level {}".format(level))
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    if (showplot):
        # Add a caption
        plt.figtext(0.5, 0.02, savefig, ha='center', fontsize=10, color='gray')
        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(savefig)
    plt.close()
    # plt.show()    

def visualize_data_from_tensor(X_train_bin, y_train, savefig, showplot=False):
    X_train_embedded = X_train_bin.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    y_train_input = y_train.reshape(X_train_bin.shape[0], -1) / 10.0
    
    umap = UMAP()
    # Fit and transform the data
    X_train_umap = umap.fit_transform(X_train_embedded)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap="Spectral", s=1, alpha=0.6)
    plt.colorbar(scatter, label="Digit Label")
    plt.title("UMAP Encoding")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    if (showplot):
        # Add a caption
        plt.figtext(0.5, 0.02, savefig, ha='center', fontsize=10, color='gray')
        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(savefig)
    plt.close()
  

def project_points_to_simplex(points):
    """
    Projects each point in a multidimensional cube [0, 1]^n onto the n-dimensional simplex.

    Parameters:
        points (np.ndarray): A 2D numpy array where each row is a point in the hypercube [0, 1]^n.

    Returns:
        np.ndarray: A 2D numpy array with each row projected onto the n-dimensional simplex.
    """
    # Number of points and dimension
    num_points, dim = points.shape

    # Array to store the projected points
    projected_points = np.zeros_like(points)

    for i in range(num_points):
        point = points[i]
        
        # Step 1: Sort the point in descending order
        u = np.sort(point)[::-1]
        
        # Step 2: Find the largest k such that the projection condition holds
        cumulative_sum = np.cumsum(u)
        rho = np.where(u > (cumulative_sum - 1) / (np.arange(dim) + 1))[0][-1]
        
        # Step 3: Compute theta
        theta = (cumulative_sum[rho] - 1) / (rho + 1)
        
        # Step 4: Project point onto the simplex
        projected_points[i] = np.maximum(point - theta, 0)

    return projected_points


def load_train_data(file_name, cell2id, drug2id):
    """
    Load the cell drug response data from a file and return feature vectors and labels.

    Parameters:
        file_name (str): the name of the file containing cell line, drug, and response values
        cell2id (dict): a dictionary mapping cell line names to unique identifiers
        drug2id (dict): a dictionary mapping drug names to unique identifiers
    
    Returns: 
        [list1, list2]: a list of feature vectors and a list of labels
    """
     
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')

            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            label.append([float(tokens[2])])

    return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):
    """
    Prepare the test data for prediction.
    
    Parameters:
        test_file (str): the name of the file containing cell line, drug, and response values
        cell2id_mapping_file (str): the name of the file containing the mapping of cell lines to unique identifiers
        drug2id_mapping_file (str): the name of the file containing the mapping of drugs to unique identifiers
        
    Returns:
        [(torch.Tensor, torch.Tensor)]: a tuple containing the test feature vectors, the test labels, the cell line mapping, and the drug mapping
    """
    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping


def load_mapping(mapping_file):
    """
    From the name index file, create a dictionary mapping names to unique identifiers.

    Parameters:
        mapping_file (str): the name of the file containing the mapping of names to unique identifiers
    
    Returns:
        dict: a dictionary mapping names to unique identifiers
    """
    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()

    return mapping


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):
    """
    Prepare the training data for prediction and the testing data.

    Parameters:
        train_file (str): the name of the file containing the training data
        test_file (str): the name of the file containing the testing data
        cell2id_mapping_file (str): the name of the file containing the mapping of cell lines to unique identifiers
        drug2id_mapping_file (str): the name of the file containing the mapping of drugs to unique identifiers
    
    Returns:
        [(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)]: a tuple containing the training feature vectors, the training labels, the testing feature vectors, and the testing labels
    """
    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label)), cell2id_mapping, drug2id_mapping


def build_input_vector(input_data, cell_features, drug_features):
    """
    Combine the cell and drug features into a single input vector.

    Parameters:
        input_data (torch.Tensor): a tensor containing the indices of the cell lines and drugs
        cell_features (np.ndarray): a numpy array containing the cell features
        drug_features (np.ndarray): a numpy array containing the drug features
    
    Returns:
        torch.Tensor: a tensor containing the combined cell and drug features
    """
    genedim = len(cell_features[0,:])
    drugdim = len(drug_features[0,:])
    feature = np.zeros((input_data.size()[0], (genedim+drugdim)))

    for i in range(input_data.size()[0]):
        feature[i] = np.concatenate((cell_features[int(input_data[i,0])], drug_features[int(input_data[i,1])]), axis=None)

    feature = torch.from_numpy(feature).float()
    return feature


def load_feature(feature_file):
    """
    Load the feature vectors from a file.

    Parameters:
        feature_file (str): the name of the file containing the feature vectors
    
    Returns:
        np.ndarray: a numpy array containing the feature vectors
    """
    feature = np.loadtxt(feature_file, delimiter=",", dtype=int)
    return feature


class CSVDrugResponseDataset(Dataset):
    def __init__(self, data_dir, data_cat, small_data = True, mutation_truncate_length = 3008):
        path_head = data_dir + "/cell_drug_responses_"
        if (small_data):
            path_tail = "_small.csv"
        else:
            path_tail = ".csv"
        assert data_cat in ["training", "testing"], "data_cat must be either 'training' or 'testing'."
        
        self.responses_path = path_head + data_cat + path_tail
        self.mutations_path = data_dir + "/binary_mutations.csv"
        self.morgan_footprints_path = data_dir + "/morgan_footprints.csv"
        
        # Read only the header to get column names
        with open(self.responses_path, "r") as f:
            header = f.readline().strip().split(",")
        
        mutations_ref = pd.read_csv(self.mutations_path, nrows=mutation_truncate_length)
        self.mutations_dict = {str(ccl): np.array(mutations_ref[ccl]).astype(np.uint8) for ccl in mutations_ref}

        morgan_footprints_ref = pd.read_csv(self.morgan_footprints_path)
        self.morgan_footprints_dict = {str(cpd): np.array(morgan_footprints_ref[cpd]).astype(np.uint8) for cpd in morgan_footprints_ref}

        self.label_col = header[-1]  # Last column name (AUC)
        self.feature_cols = header[:-1]  # All except last column
        
        # Read only labels to get dataset length
        self.data_info = pd.read_csv(self.responses_path, usecols=[self.label_col])
        self.length = len(self.data_info)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Read a single row efficiently
        ccl, cpd, auc = pd.read_csv(self.responses_path, skiprows=idx+1, nrows=1, header=None).values[0]
        mutation = torch.from_numpy(self.mutations_dict[str(ccl)]).to(torch.float32)
        morgan_footprint = torch.from_numpy(self.morgan_footprints_dict[str(cpd)]).to(torch.float32)
        features = torch.cat((mutation, morgan_footprint))  # Features (all except last column)
        label = torch.tensor(auc, dtype=torch.float32)  # Last column (AUC)
        return features, label