import pandas as pd
import numpy as np
import torch

from scipy.sparse import csr_matrix, vstack, hstack
from torch.utils.data import Dataset, DataLoader


class CSRSparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # Feature tensor
        self.y = y  # Label tensor
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
def extract_sparse_vectors(df, column_names):
    data_list = []
    for col in column_names:
        if col in df.columns:
            data_list.append(csr_matrix(df[col].values.reshape(1, -1)).astype(np.uint8))
        else:
            # Handle missing columns by appending zero vectors
            data_list.append(csr_matrix(np.zeros((df.shape[0], 1), dtype=np.uint8)))
    return vstack(data_list, format="csr")


def load_data(mut_len: int, data_size: int, batch_size: int, dir: str = "../dataset/Cancer/") -> DataLoader:
    '''
    Load mutation, drug and response data from the dataset
    '''
    # Cell line structure data frame, columns are cell lines, rows are gene mutations ordered by descending frequency
    mutation_df = pd.read_csv(dir+"binary_mutations.csv")
    mutation_df = mutation_df.head(mut_len)
    # Response data frame, columns have cell line name, drug name and auc score
    response_df = pd.read_csv(dir+"cell_drug_responses.csv")
    response_df = response_df.sample(n=data_size) 
    # Drug structure data frame, columns are drug name, rows are morgan footprint entries
    morgan_footprints_df = pd.read_csv(dir+"morgan_footprints.csv")

    # 1. Extract mutation vectors (batch processing)
    mutation_vectors = extract_sparse_vectors(mutation_df, response_df['ccl_name'])

    # 2. Extract Morgan fingerprint vectors (batch processing)
    fingerprint_vectors = extract_sparse_vectors(morgan_footprints_df, response_df['cpd_name'])

    # 3. Extract AUC labels
    labels = csr_matrix(response_df['normalized_auc'].values.reshape(-1, 1)).toarray().flatten()

    # 4. Concatenate sparse matrices to form feature matrix
    features = hstack([mutation_vectors, fingerprint_vectors], format="csr").toarray()

    # 6. Convert to PyTorch Tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # 7. Create dataset and dataloader
    dataset = CSRSparseDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


