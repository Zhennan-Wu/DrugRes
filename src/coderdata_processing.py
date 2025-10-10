
import os
from os import name
import pandas as pd
import coderdata as cd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # Correct import for new RDKit
import pubchempy as pcp
import matplotlib.pyplot as plt


def visualize_response_statistics(dataset: cd.Dataset, data_name, metric: str = 'aac'):
    """
    Visualize summary statistics and distribution of AAC values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the AAC values.
    column : str, default='aac'
        The name of the column containing AAC values.
    """
    df = dataset.format(data_type='experiments', metrics=metric)
    column = 'dose_response_value'
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Drop NaNs to avoid plotting issues
    data = pd.to_numeric(df[column], errors='coerce').dropna()

    if data.empty:
        print(f"⚠️ Column '{metric}' in dataset is empty or non-numeric after cleaning.")
        return
    data = data[np.isfinite(data)]
    # lower, upper = data.quantile(0.01), data.quantile(0.99)
    # data = data.clip(lower, upper)
    # Compute summary statistics
    summary = data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"📊 Summary statistics for `{metric}`:\n")
    print(summary.to_string())
    print()

    # Set up plotting layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram + KDE
    axes[0].hist(data, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='black')
    data.plot(kind='kde', ax=axes[0], color='darkred')
    axes[0].set_title(f'Histogram & KDE of {metric}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Density')

    # Boxplot
    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='darkred'))
    axes[1].set_title(f'Boxplot of {metric}')
    axes[1].set_ylabel(column)

    plt.tight_layout()
    file_dir = f"../plts/response_metrics/{data_name}"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(f"{file_dir}/{metric}_distribution.png")
    plt.close(fig)
    print(f"Distribution plot saved to {file_dir}/{metric}_distribution.png")


def summarize_all_available_metrics(datasets: dict):
    """
    Summarize all available metrics in the experiments DataFrame of a Dataset.

    Parameters
    ----------
    dataset : cd.Dataset
        The Dataset object containing the experiments DataFrame.
    """
    avail_metrics = set()
    for name, data in datasets.items():
        if (not hasattr(data, 'experiments')) or (data.experiments is None):
            print(f"Experiments data not found for dataset '{name}'. Skipping.")
            continue
        metrics = data.experiments.dose_response_metric.unique()
        avail_metrics.update(metrics)
        print(f"Available metrics in dataset '{name}': {metrics}")
    common_metrics = set(avail_metrics)
    for name, data in datasets.items():
        if (not hasattr(data, 'experiments')) or (data.experiments is None):
            continue
        metrics = data.experiments.dose_response_metric.unique()
        common_metrics.intersection_update(set(metrics))
    return list(common_metrics)


def visualize_responses_across_datasets(dfs: dict, metric: str = 'aac'):
    """
    Visualize summary statistics and distribution of AAC values across multiple datasets.

    Parameters
    ----------
    dfs : dict
        A dictionary where keys are dataset names and values are DataFrames.
    metric : str, default='aac'
        The name of the metric to visualize.
    """
    for name, df in dfs.items():
        print(f"Dataset: {name}")
        visualize_response_statistics(df, name, metric=metric)


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize a DataFrame with detailed column-wise statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.

    Returns
    -------
    summary : pd.DataFrame
        A summary table with columns:
        ['dtype', 'num_null', 'pct_null', 'num_inf', 'num_unique',
         'min', 'max', 'example_values']
    """
    n_rows = len(df)
    summary_data = []

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        # Handle nulls & inf
        num_nan = series.isna().sum()
        num_inf = np.isinf(series).sum() if pd.api.types.is_numeric_dtype(series) else 0
        num_null_total = num_nan + num_inf
        pct_null = (num_null_total / n_rows * 100) if n_rows > 0 else 0

        # Handle unique count (including NaN)
        num_unique = series.nunique(dropna=False)

        # Handle min/max only for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            cleaned = series.replace([np.inf, -np.inf], np.nan)
            min_val = cleaned.min(skipna=True)
            max_val = cleaned.max(skipna=True)
        else:
            min_val = None
            max_val = None

        # Example values (to quickly inspect categories or strings)
        example_vals = series.dropna().unique()[:5]
        example_str = ", ".join(map(str, example_vals))

        summary_data.append({
            'column': col,
            'dtype': dtype,
            'num_null': num_null_total,
            'pct_null': round(pct_null, 2),
            'num_inf': num_inf,
            'num_unique': num_unique,
            'min': min_val,
            'max': max_val,
            'example_values': example_str
        })

    summary = pd.DataFrame(summary_data).set_index('column')
    return summary


def summarize_datasets(dfs: dict):
    """
    Summarize multiple DataFrames by checking null values and value ranges per column.

    Parameters
    ----------
    dfs : dict
        A dictionary where keys are dataset names and values are DataFrames.

    Returns
    -------
    summaries : dict
        A dictionary where keys are dataset names and values are summary DataFrames.
    """
    summaries = {}
    for name, df in dfs.items():
        summaries[name] = {}
        for feature in df.types():
            summaries[name][feature] = summarize_dataframe(getattr(df, feature))
    return summaries


def summarize_df_matrices(dfs: dict):
    """
    Summarize multiple DataFrames by checking null values and value ranges per column.

    Parameters
    ----------
    dfs : dict
        A dictionary where keys are dataset names and values are DataFrames.

    Returns
    -------
    summaries : dict
        A dictionary where keys are dataset names and values are summary DataFrames.
    """
    summaries = {}
    for feature, df in dfs.items():
        summaries[feature] = {}
        for dataset, value in df.items():
            summaries[feature][dataset] = summarize_dataframe(value)
    return summaries


def unify_feature_across_dataset(dfs: dict, feature: str, *args, **kwargs):
    """
    Get all columns from DataFrames return by the Dataset feature
    Args:
        dfs (dict): A dictionary where keys are dataset names and values are Dataset.
        feature (str): The feature type to ex
    """
    all_columns = set()
    df_dict_aligned = {}
    if (feature == "transcriptomics"):
        sample_feature_dfs = {}
        for name, dataset in dfs.items():
            df = dataset.format(data_type="transcriptomics")
            all_columns.update(df.columns)
            sample_feature_dfs[name] = df
        df_dict_aligned = {
            name: df.reindex(columns=all_columns)
            for name, df in sample_feature_dfs.items()
        }
    elif (feature == "mutations"):
        if "mutation_type" not in kwargs:
            raise ValueError("mutation_type must be specified for mutations feature")
        sample_feature_dfs = {}
        for name, dataset in dfs.items():
            df = dataset.format(data_type="mutations", mutation_type=kwargs.get("mutation_type", None))
            all_columns.update(df.columns)
            sample_feature_dfs[name] = df
        df_dict_aligned = {
            name: df.reindex(columns=all_columns)
            for name, df in sample_feature_dfs.items()
        }
    elif (feature == "proteomics"):
        sample_feature_dfs = {}
        for name, dataset in dfs.items():
            df = dataset.format(data_type="proteomics")
            all_columns.update(df.columns)
            sample_feature_dfs[name] = df
        df_dict_aligned = {
            name: df.reindex(columns=all_columns)
            for name, df in sample_feature_dfs.items()
        }
    elif (feature == "copy_number"):
        sample_feature_dfs = {}
        for name, dataset in dfs.items():
            df = dataset.format(data_type="copy_number")
            all_columns.update(df.columns)
            sample_feature_dfs[name] = df
        df_dict_aligned = {
            name: df.reindex(columns=all_columns)
            for name, df in sample_feature_dfs.items()
        }
    else:
        print(f"Feature {feature} not supported")
    return df_dict_aligned


def get_cell_id_across_dfs(dfs: dict):
    cell_id_ref = {}
    for feature, df in dfs.items():
        for dataset, value in df.items():
            if dataset not in cell_id_ref:
                cell_id_ref[dataset] = {}
            cell_id_ref[dataset][feature] = value.index.tolist()
    return cell_id_ref


def get_total_mutation_types(datasets: dict):
    mutation_types = set()
    for _, dataset in datasets.items():
        types = dataset.mutations.variant_classification.unique().tolist()
        mutation_types.update(types)
    return list(mutation_types)


def data_attribute_check(datasets: dict, ignored_features: list=None):
    feature_reference = [
        'transcriptomics', 'proteomics', 'mutations',
        'copy_number', 'samples', 'drugs',
        'experiments', 'genes'
    ]
    if ignored_features is not None:
        feature_reference = [f for f in feature_reference if f not in ignored_features]
    
     # Check each dataset
    
    for dataset_name, data in datasets.items():
        # Get available feature types for this dataset
        feature_types = data.types()
        
        # Find which reference features are missing
        missing_features = [f for f in feature_reference if f not in feature_types]
        
        # Print result
        if missing_features:
            print(f"Dataset '{dataset_name}' is missing: {', '.join(missing_features)}")
        else:
            print(f"Dataset '{dataset_name}' has all required features ✅")


def min_max_scaling(df):
    """
    Apply min-max scaling to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to scale.

    Returns
    -------
    pd.DataFrame
        The min-max scaled DataFrame.
    """
    min_value = df.min().min()
    max_value = df.max().max()
    # Perform min-max scaling
    scaled_df = (df - min_value) / (max_value - min_value)
    return scaled_df


def min_max_scaling_across_dfs(df_dict, feature: str):
    """
    Apply min-max scaling across multiple DataFrames.
    
    Parameters
    ----------
    df_dict : dict
        A dictionary where keys are DataFrame names and values are DataFrames.
    feature : str
        The feature to scale.

    Returns
    -------
    dict
        A new dictionary with min-max scaled DataFrames.
    """
    if (feature == "transcriptomics"):
        for name, df in df_dict.items():
            log_trans = np.log1p(df)
            df_dict[name] = min_max_scaling(log_trans)
    elif (feature == "proteomics"):
        for name, df in df_dict.items():
            df_dict[name] = min_max_scaling(df)
    elif (feature == "copy_number"):
        for name, df in df_dict.items():
            df_dict[name] = min_max_scaling(df)
    else:
        print(f"Feature {feature} not supported for min-max scaling")
    return df_dict


def filter_biomarkers(df_dict, data_size_ref, proportion=1e-2):
    """
    Filters out biomarkers (mutation types) with few samples.
    Args:
        df_dict (dict): A dictionary where keys are DataFrame names and values are DataFrames.
    Returns:
        dict: A new dictionary with filtered DataFrames.
    """
    filtered_dfs = {}
    for feature, df in df_dict.items():
        thres = []
        for dataset, value in df.items():
            thres.append(value.shape[0] >= proportion*data_size_ref[dataset])
        if (sum(thres) >= 0.5*len(thres)):
            filtered_dfs[feature] = df
        else:
            print(f"Feature {feature} is filtered out due to insufficient samples")

    return filtered_dfs


def filter_feature(df_dict: dict, thres: float=1e-2):
    """
    Filters out columns in DataFrames based on variance thresholds.
    
    Parameters
    ----------
    df_dict : dict
        A dictionary where keys are DataFrame names and values are DataFrames.
    thres : dict
        A dictionary where keys are DataFrame names and values are tuples (min_variance, max_variance).
    Returns
    -------
    dict
        A new dictionary with filtered DataFrames.
    """
    filtered_dfs = {}
    for feature, df in df_dict.items():
        filtered_dfs[feature] = {}
        vars_list = [dataset.var() for dataset in df.values()]
        vars_df = pd.concat(vars_list, axis=1)
        max_var_per_col = vars_df.max(axis=1)
        selected_cols = max_var_per_col[max_var_per_col > thres].index.tolist()
        print(f"Feature {feature}: selected {len(selected_cols)} out of {vars_df.shape[0]} features")
        for dataset, value in df.items():
            filtered_dfs[feature][dataset] = value[selected_cols]
    return filtered_dfs


def dip_data_across_dfs(df_dict, **kwargs):
    """
    Apply dropout to DataFrames by randomly setting a proportion of values to NaN.
    
    Parameters
    ----------
    df_dict : dict
        A dictionary where keys are DataFrame names and values are DataFrames.
    dip_size : int
        The number of values to set to NaN in each DataFrame.

    Returns
    -------
    dict
        A new dictionary with DataFrames after applying dropout.
    """
    frac = kwargs.get("fraction", 0.1)
    df_dict_dipped = {}
    for feature, df in df_dict.items():
        df_dict_dipped[feature] = {}
        for dataset, value in df.items():
            df_dict_dipped[feature][dataset] = value.sample(frac=frac, random_state=42)
    return df_dict_dipped


def dip_experiments_across_dfs(df_dict, **kwargs):
    """
    Apply dropout to DataFrames by randomly setting a proportion of values to NaN.
    
    Parameters
    ----------
    df_dict : dict
        A dictionary where keys are DataFrame names and values are DataFrames.
    dip_size : int
        The number of values to set to NaN in each DataFrame.

    Returns
    -------
    dict
        A new dictionary with DataFrames after applying dropout.
    """
    frac = kwargs.get("fraction", 0.1)
    random_state = kwargs.get("random_state", 42)
    df_dict_dipped = {}
    for name, value in df_dict.items():
        df_dict_dipped[name] = value.sample(frac=frac, random_state=random_state)
    return df_dict_dipped


def extract_row(df_ref, sample_id):
    """
    Extract a row from a DataFrame based on sample_id.
    If sample_id is not found, return a zero vector of the same length as the number of columns in df_ref.
    Parameters
    ----------
    df_ref : pd.DataFrame
        DataFrame from which to extract the row.
    sample_id : str
        The sample ID to look for in the DataFrame index.
    Returns
    -------
    feature : np.ndarray
        The extracted row as a numpy array, or a zero vector if sample_id is not found
    id : str or int
        The sample_id if found, otherwise -1
    """
    if sample_id in df_ref.index:
        feature = df_ref.loc[sample_id].to_numpy()
        feature[np.isnan(feature)] = 0
        id = sample_id
    else:
        feature = np.zeros((df_ref.shape[1],))   
        id = -1
    return feature, id


def repre_drug(drug_ids, drug_embeds, d_id):
    """
    Represent a drug by its embedding.
    Parameters
    ----------
    drug_ids : list
        List of drug IDs corresponding to the drug embeddings.
    drug_embeds : np.ndarray
        Array of drug embeddings.
    d_id : str or int
        The drug ID to look for.
    Returns
    -------
    embed : np.ndarray
        The embedding of the drug.
    """
    d_idx = np.where(drug_ids == d_id)
    embed = drug_embeds[d_idx][0]
    return embed


def preprocess_experiment(experiments, drug_ids, drug_embeds, filtered_dfs, drug_target:str=None, show_status:bool=True):
    """
    Preprocess experiment data by extracting features, drug representations, and responses.
    Parameters
    ----------
    experiments : pd.DataFrame
        DataFrame containing experiment data with columns 'improve_sample_id', 'improve_drug_id', and 'dose_response_value'.
    drug_ids : list
        List of drug IDs corresponding to the drug embeddings.
    drug_embeds : np.ndarray
        Array of drug embeddings.
    filtered_dfs : dict
        Dictionary of DataFrames containing features, keyed by feature type.
    drug_target : str, optional
        Specific drug ID to filter experiments. If None, all drugs are processed.
    show_status : bool, optional
        Whether to print status updates during processing.
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


def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to molecule
    if mol is None:
        return None  # Invalid SMILES
    
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)  # New recommended method
    fingerprint = generator.GetFingerprint(mol)

    # Convert to NumPy array
    fingerprint_bits = np.array(fingerprint, dtype=int)

    return fingerprint_bits


def get_drug_fingerprints(drug_id, drugs_ref):
    """
    Get the Morgan fingerprint for a drug given its ID.
    Parameters
    ----------
    drug_id : str or int
        The drug ID to look for.
    drugs_ref : pd.DataFrame
        DataFrame containing drug reference data with columns 'improve_drug_id' and 'canSMILES'.
    Returns
    -------
    np.ndarray
        The Morgan fingerprint of the drug as a numpy array.
    """
    smiles = drugs_ref[drugs_ref["improve_drug_id"]==drug_id]["canSMILES"].values[0]
    fingerprint = get_morgan_fingerprint(smiles)
    if fingerprint is None:
        print(f"Invalid SMILES for drug ID {drug_id}")
        return None
    return np.array(fingerprint)


def preprocess_experiment_with_footprint(experiments_dfs, drugs_ref_dfs, filtered_dfs, drug_target:str=None, show_status:bool=True):
    """
    Preprocess experiment data by extracting features, drug fingerprints, and responses.
    Parameters
    ----------
    experiments : pd.DataFrame
        DataFrame containing experiment data with columns 'improve_sample_id', 'improve_drug_id', and 'dose_response_value'.
    drugs_ref : pd.DataFrame
        DataFrame containing drug reference data.
    filtered_dfs : dict
        Dictionary of DataFrames containing features, keyed by feature type.
    drug_target : str, optional
        Specific drug ID to filter experiments. If None, all drugs are processed.
    show_status : bool, optional
        Whether to print status updates during processing.
    """
    feature_dict = {}
    response_dict = {}
    
    for name, experiments in experiments_dfs.items():
        feature_dict[name] = {}
        response_dict[name] = []
        drugs_ref = drugs_ref_dfs[name]
        print(f"Processing dataset: {name} with {len(experiments)} experiments")
        for index, row in experiments.iterrows():
            if drug_target is None or str(row["improve_drug_id"]) == str(drug_target):
                footprint = get_drug_fingerprints(row["improve_drug_id"], drugs_ref)
                if "fingerprint" not in feature_dict[name]:
                    feature_dict[name]["fingerprint"] = []
                if footprint is None:
                    print(f"Skipping experiment {index} with drug ID {row['improve_drug_id']} due to invalid SMILES")

                    continue
                
                feature_dict[name]["fingerprint"].append(footprint)

                improve_sample_id = row["improve_sample_id"]
                if "improve_sample_id" not in feature_dict[name]:
                    feature_dict[name]["improve_sample_id"] = []
                feature_dict[name]["improve_sample_id"].append(improve_sample_id)
                for key, df in filtered_dfs.items():
                    value = df[name]
                    feature, id = extract_row(value, improve_sample_id)
                    if key not in feature_dict[name]:
                        feature_dict[name][key] = []
                    feature_dict[name][key].append(feature)

                response_dict[name].append(row["dose_response_value"])
        print(f"Finished processing dataset: {name} with {len(response_dict[name])} selected samples")
    return feature_dict, response_dict


def extract_data_position(feature_list, features, cell_id_ref):
    """
    Extract indices of samples that have all specified features available.
    Parameters
    ----------
    feature_list : list
        List of feature types to check.
    features : dict
        Dictionary of features, where keys are feature types and values are numpy arrays.
    cell_id_ref : dict
        Dictionary mapping feature types to lists of sample IDs.
    Returns
    -------
    list
        List of indices of samples that have all specified features.
    """
    cell_ids = set(cell_id_ref[feature_list[0]])
    for feature in feature_list[1:]:
        cell_ids = cell_ids.intersection(set(cell_id_ref[feature]))
    samples = features["improve_sample_id"]
    indices = [i for i, val in enumerate(samples) if val in cell_ids]
    return indices


def cat_drug_response_features_across_datasets(features_dfs:dict, labels_dfs: dict, cell_id_ref_dfs: dict, feature_list: list) -> np.ndarray:
    """
    Concatenate features from different omics data types across multiple datasets.
    Parameters
    ----------
    features : dict
        Dictionary of features, where keys are dataset names and values are dictionaries of feature types and numpy arrays.
    labels : dict
        Dictionary where keys are dataset names and values are lists of labels corresponding to the samples.
    cell_id_ref : dict
        Dictionary mapping feature types to lists of sample IDs.
    feature_list : list
        List of feature types to concatenate.
    Returns
    -------
    np.ndarray
        Concatenated feature array and corresponding labels.
    """
    training_datas = []
    training_labels = []
    for name, features in features_dfs.items():
        labels = labels_dfs[name]
        cell_id_ref = cell_id_ref_dfs[name]
        data, label = cat_drug_response_features(features, labels, cell_id_ref, feature_list)
        training_datas.append(data)
        training_labels.append(label)
    return np.concatenate(training_datas, axis=0), np.concatenate(training_labels, axis=0)


def cat_drug_response_features(features:dict, labels: list, cell_id_ref: dict, feature_list: list) -> np.ndarray:
    """
    Concatenate features from different omics data types.
    Parameters
    ----------
    features : dict
        Dictionary of features, where keys are feature types and values are numpy arrays.
    labels : list
        List of labels corresponding to the samples.
    cell_id_ref : dict
        Dictionary mapping feature types to lists of sample IDs.
    feature_list : list
        List of feature types to concatenate.
    Returns
    -------
    np.ndarray
        Concatenated feature array and corresponding labels.
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


def cat_cell_features(features:dict, sample_size: int, feature_list: list=None) -> np.ndarray:
    """
    Concatenate features from different omics data types.
    Parameters
    ----------
    features : dict
        Dictionary of features, where keys are feature types and values are numpy arrays.
    sample_size : int
        Number of samples to process.
    feature_list : list, optional
        List of feature types to concatenate. If None, all features in the dictionary are used.
    Returns
    -------
    np.ndarray
        Concatenated feature array.
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


def preprocess_cell_line(cell_id_ref, filtered_dfs, feature_list):
    """
    Preprocess cell line data by extracting and concatenating features from multiple DataFrames.
    Parameters
    ----------
    cell_id_ref : dict
        Dictionary mapping feature types to lists of sample IDs.
    filtered_dfs : dict
        Dictionary of DataFrames containing features, keyed by feature type.
    feature_list : list
        List of feature types to concatenate.
    Returns
    ------- 
    np.ndarray
        Concatenated feature array for cell lines.
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