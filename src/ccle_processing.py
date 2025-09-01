import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # Correct import for new RDKit
import pubchempy as pcp


def data_statistics(df: pd.DataFrame) -> None:
    """
    Prints basic statistics of the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values[missing_values > 0])
    
    # Check for data range
    min_value = df.min().min()
    max_value = df.max().max()
    print(f"Min: {min_value}, Max: {max_value}")


def min_max_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies min-max scaling to the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to scale.
    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    min_value = df.min().min()
    max_value = df.max().max()
    # Perform min-max scaling
    scaled_df = (df - min_value) / (max_value - min_value)
    return scaled_df


def align_dfs(df_dict: dict) -> dict:
    """
    Aligns multiple DataFrames to have the same columns by reindexing them.
    Args:
        df_dict (dict): A dictionary where keys are DataFrame names and values are DataFrames.
    Returns:
        dict: A new dictionary with aligned DataFrames.
    """
    # Step 1: Get the union of all columns
    all_columns = set()
    for df in df_dict.values():
        all_columns.update(df.columns)
    all_columns = sorted(all_columns)  # Optional: keep column order consistent

    # Step 2: Reindex each DataFrame to have the full set of columns
    df_dict_aligned = {
        name: df.reindex(columns=all_columns)
        for name, df in df_dict.items()
    }
    return df_dict_aligned


def filter_feature(df_dict: dict, thres: dict, aligned:bool=True) -> dict:
    """
    Filters features in each DataFrame based on variance thresholds.
    Args:
        df_dict (dict): A dictionary where keys are DataFrame names and values are DataFrames.
        thres (dict): A dictionary of tuples with variance thresholds for each DataFrame.
        aligned (bool): If True, align the DataFrames to have the same columns.
    Returns:
        dict: A new dictionary with filtered DataFrames.
    """
    if (aligned):
        all_columns = set()
        for key, df in df_dict.items():
            variances = df.var()
            # Filter column names with variance > 0 and < 10
            selected_cols = variances[(variances > thres[key][0]) & (variances < thres[key][1])].index
            all_columns.update(selected_cols)
        all_columns = sorted(all_columns)  # Optional: keep column order consistent
        df_dict_aligned = {
            name: df.reindex(columns=all_columns)
            for name, df in df_dict.items()
        }
    else:
        df_dict_aligned = {}
        for key, df in df_dict.items():
            variances = df.var()
            # Filter column names with variance > 0 and < 10
            selected_cols = variances[(variances > thres[key][0]) & (variances < thres[key][1])].index
            df_dict_aligned[key] = df[selected_cols]
            
    return df_dict_aligned


def filter_biomarkers(df_dict: dict, proportion:float=0.5) -> dict:
    """
    Filters out biomarkers with low variance across samples.
    Args:
        df_dict (dict): A dictionary where keys are DataFrame names and values are DataFrames.
    Returns:
        dict: A new dictionary with filtered DataFrames.
    """
    filtered_dfs = {}
    sample_sizes = []
    for name, df in df_dict.items():
        sample_sizes.append(df.shape[0])
    max_sample_size = max(sample_sizes)
    min_sample_size = min(sample_sizes)
    threshold = proportion * (max_sample_size - min_sample_size) + min_sample_size
    print(f"Threshold: {threshold}")
    
    for name, df in df_dict.items():
        if (df.shape[0] > threshold):
            filtered_dfs[name] = df
    return filtered_dfs


def cat_cell_features(features:dict, sample_size: int, feature_list: list=None) -> np.ndarray:
    """
    Concatenate features from different omics data types.
    """
    training_data = []
    for s_id in range(sample_size):
        cell_features = []
        if feature_list is not None:
            for feature in feature_list:
                if feature in features:
                    cell_features.append(features[feature][s_id])
                else:
                    raise KeyError(f"Feature '{feature}' not found in features dictionary.")
        else:
            for value in features.values():
                cell_features.append(value[s_id])
        training_data.append(np.concatenate(cell_features, axis=0))
    return np.array(training_data)


def extract_row(df_ref: pd.DataFrame, sample_id: str) -> tuple:
    """
    Extracts a row from the DataFrame based on the sample ID.
    Args:
        df_ref (pd.DataFrame): The DataFrame to extract from.
        sample_id (str): The sample ID to look for.
    Returns:
        tuple: A tuple containing the feature vector and the sample ID.
    """
    if sample_id in df_ref.index:
        feature = df_ref.loc[sample_id].to_numpy()
        feature[np.isnan(feature)] = 0
        id = sample_id
    else:
        feature = np.zeros((df_ref.shape[1],))   
        id = -1
    return feature, id


def preprocess_cell_line(cell_id_ref: dict, filtered_dfs: pd.DataFrame, feature_list: list) -> np.ndarray:
    """
    Preprocess cell line data by filtering and concatenating features.
    Args:
        cell_id_ref (dict): A dictionary mapping features to cell IDs.
        filtered_dfs (pd.DataFrame): A DataFrame containing the filtered features.
        feature_list (list): A list of features to include.
    Returns:
        np.ndarray: A NumPy array containing the concatenated features.
    """
    feature_dict = {}
    cell_ids = set(cell_id_ref[feature_list[0]])
    for feature in feature_list[1:]:
        cell_ids = cell_ids.intersection(set(cell_id_ref[feature]))
    for improve_sample_id in cell_ids:
        improve_sample_id = int(improve_sample_id)
        for key, value in filtered_dfs.items():
            feature, _ = extract_row(value, improve_sample_id)
            if key not in feature_dict:
                feature_dict[key] = []
            feature_dict[key].append(feature)

    X = cat_cell_features(feature_dict, len(list(cell_ids)), feature_list)
    print(f"Cell line count: {len(cell_ids)}")
    return X


def get_smiles(drug_name: str) -> str:
    """
    Fetches the SMILES representation of a drug using PubChemPy.
    Args:
        drug_name (str): The name of the drug.
    Returns:
        str: The SMILES representation of the drug.
    """
    compounds = pcp.get_compounds(drug_name, 'name')
    if compounds:
        return compounds[0].isomeric_smiles  # Returns the first match's SMILES
    return None  # Return None if no results found


def get_morgan_fingerprint(smiles: str, radius: int=2, n_bits:int=2048) -> np.ndarray:
    """
    Generates a Morgan fingerprint from a SMILES string.
    Args:
        smiles (str): The SMILES representation of the molecule.
        radius (int): The radius for the Morgan fingerprint.
        n_bits (int): The size of the fingerprint.
    Returns:
        np.ndarray: The Morgan fingerprint as a NumPy array.
    """
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to molecule
    if mol is None:
        return None  # Invalid SMILES
    
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)  # New recommended method
    fingerprint = generator.GetFingerprint(mol)

    # Convert to NumPy array
    fingerprint_bits = np.array(fingerprint, dtype=int)

    return fingerprint_bits


def get_drug_fingerprints(drug_id: str, drugs_ref: pd.DataFrame) -> np.ndarray:
    """
    Generates a Morgan fingerprint for a drug based on its ID.
    Args:
        drug_id (str): The ID of the drug.
        drugs_ref (pd.DataFrame): DataFrame containing drug information.
    Returns:
        np.ndarray: The Morgan fingerprint as a NumPy array.
    """
    smiles = drugs_ref[drugs_ref["improve_drug_id"]==drug_id]["canSMILES"].values[0]
    fingerprint = get_morgan_fingerprint(smiles)
    if fingerprint is None:
        raise ValueError(f"Invalid SMILES for drug ID {drug_id}")
    return np.array(fingerprint)


def preprocess_experiment_with_footprint(experiments: pd.DataFrame, drugs_ref: pd.DataFrame, filtered_dfs:pd.DataFrame, drug_target:str=None, show_status:bool=True) -> tuple:
    """
    Preprocesses experiment data by filtering and extracting features.
    Args:
        experiments (pd.DataFrame): DataFrame containing experiment data.
        drugs_ref (pd.DataFrame): DataFrame containing drug information.
        filtered_dfs (pd.DataFrame): DataFrame containing filtered features.
        drug_target (str): Optional drug target to filter by.
        show_status (bool): If True, shows processing status.
    Returns:
        tuple: A tuple containing a dictionary of features and a list of response values.
    """
    feature_dict = {}
    resp = []
    for index, row in experiments.iterrows():
        if drug_target is None or str(row["improve_drug_id"]) == str(drug_target):
            improve_sample_id = row["improve_sample_id"]
            if "improve_sample_id" not in feature_dict:
                feature_dict["improve_sample_id"] = []
            feature_dict["improve_sample_id"].append(improve_sample_id)
            for key, value in filtered_dfs.items():
                feature, id = extract_row(value, improve_sample_id)
                if key not in feature_dict:
                    feature_dict[key] = []
                feature_dict[key].append(feature)

            footprint = get_drug_fingerprints(row["improve_drug_id"], drugs_ref)
            if "fingerprint" not in feature_dict:
                feature_dict["fingerprint"] = []
            feature_dict["fingerprint"].append(footprint)

            resp.append(row["dose_response_value"])
        if index % 1000 == 0 and show_status:
            print(f"Processed {index+1} samples")
            print(f"Selected {len(resp)} samples")
    return feature_dict, resp


def extract_data_position(feature_list: list, features: pd.DataFrame, cell_id_ref: dict) -> list:
    """
    Extracts the indices of samples that are present in all features.
    Args:
        feature_list (list): A list of features to check.
        features (pd.DataFrame): DataFrame containing the features.
        cell_id_ref (dict): A dictionary mapping features to cell IDs.
    Returns:
        list: A list of indices of samples that are present in all features.
    """
    # Initialize cell_ids with the first feature's IDs
    cell_ids = set(cell_id_ref[feature_list[0]])
    for feature in feature_list[1:]:
        cell_ids = cell_ids.intersection(set(cell_id_ref[feature]))
    samples = features["improve_sample_id"]
    indices = [i for i, val in enumerate(samples) if val in cell_ids]
    return indices


def cat_drug_response_features(features:dict, labels: list, cell_id_ref: dict, feature_list: list) -> np.ndarray:
    """
    Concatenate features from different omics data types.
    """
    training_data = []
    training_label = []
    sample_indices = extract_data_position(feature_list, features, cell_id_ref)
    for s_id in sample_indices:
        training_label.append(labels[s_id])
        cell_features = []
        for feature in feature_list:
            if feature in features:
                cell_features.append(features[feature][s_id])
            else:
                raise KeyError(f"Feature '{feature}' not found in features dictionary.")
        cell_features.append(features["fingerprint"][s_id])
        training_data.append(np.concatenate(cell_features, axis=0))
    return np.array(training_data), np.array(training_label)


def repre_drug(drug_ids: list, drug_embeds: list, d_id: str) -> np.ndarray:
    """
    Represents a drug by its ID using its embedding.
    Args:
        drug_ids (list): List of drug IDs.
        drug_embeds (list): List of drug embeddings.
        d_id (str): The drug ID to represent.
    Returns:
        np.ndarray: The embedding of the drug.
    """
    d_idx = np.where(drug_ids == d_id)
    embed = drug_embeds[d_idx][0]
    return embed


def preprocess_experiment(experiments: pd.DataFrame, drug_ids:list, drug_embeds:list, filtered_dfs:pd.DataFrame, drug_target:str=None, show_status:bool=True) -> tuple:
    """
    Preprocesses experiment data by filtering and extracting features.
    Args:
        experiments (pd.DataFrame): DataFrame containing experiment data.
        drug_ids (list): List of drug IDs.
        drug_embeds (list): List of drug embeddings.
        filtered_dfs (pd.DataFrame): DataFrame containing filtered features.
        drug_target (str): Optional drug target to filter by.
        show_status (bool): If True, shows processing status.
    Returns:
        tuple: A tuple containing a dictionary of features and a list of response values.
    """
    sample_ids_dict = {}
    feature_dict = {}
    drugf = []
    resp = []
    for index, row in experiments.iterrows():
        if drug_target is None or str(row["improve_drug_id"]) == str(drug_target):
            improve_sample_id = row["improve_sample_id"]
            for key, value in filtered_dfs.items():
                feature, id = extract_row(value, improve_sample_id)
                if key not in sample_ids_dict:
                    sample_ids_dict[key] = []
                if key not in feature_dict:
                    feature_dict[key] = []
                feature_dict[key].append(feature)
                sample_ids_dict[key].append(id)

            drug_id = row["improve_drug_id"]
            drug = repre_drug(drug_ids, drug_embeds, drug_id)
            drugf.append(drug)

            resp.append(row["dose_response_value"])
        if index % 1000 == 0 and show_status:
            print(f"Processed {index+1} samples")
            print(f"Selected {len(resp)} samples")
    return feature_dict, sample_ids_dict, drugf, resp