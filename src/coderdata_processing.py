import numbers
import os
from os import name
import torch
import pandas as pd
import coderdata as cd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # Correct import for new RDKit
import pubchempy as pcp
import matplotlib.pyplot as plt
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


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
    df = df.copy()
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


def visualize_sample_response_statistics(dataset: dict):
    """
    Visualize summary statistics and distribution of AAC values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the AAC values.
    column : str, default='aac'
        The name of the column containing AAC values.
    """
    combined_df = pd.concat(dataset.values(), keys=dataset.keys(), names=['Source', 'Row'])
    combined_df = combined_df.reset_index(level='Source').reset_index(drop=True)
    column = 'dose_response_value'

    # Drop NaNs to avoid plotting issues
    combined_df = combined_df.copy()
    data = pd.to_numeric(combined_df[column], errors='coerce').dropna()

    data = data[np.isfinite(data)]
    # lower, upper = data.quantile(0.01), data.quantile(0.99)
    # data = data.clip(lower, upper)
    # Compute summary statistics
    summary = data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"📊 Summary statistics:\n")
    print(summary.to_string())
    print()

    # Set up plotting layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram + KDE
    axes[0].hist(data, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='black')
    data.plot(kind='kde', ax=axes[0], color='darkred')
    axes[0].set_title(f'Histogram & KDE')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Density')

    # Boxplot
    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='darkred'))
    axes[1].set_title(f'Boxplot')
    axes[1].set_ylabel(column)

    plt.tight_layout()
    plt.show()


def clean_experiments_across_dfs(dfs: dict, metric: str = 'aac'):
    """
    Clean experiments DataFrames by removing rows with NaN or infinite values in the specified metric column.

    Parameters
    ----------
    dfs : dict
        A dictionary where keys are dataset names and values are DataFrames.
    metric : str, default='aac'
        The name of the metric column to clean.
    
    Returns
    -------
    dict
        A new dictionary with cleaned DataFrames.
    """
    if (metric in ["auc", "fit_auc"]):
        thres = 1.
    elif (metric in ["aac", "fit_aac"]):
        thres=0.
    else:
        raise ValueError(f"Metric '{metric}' not recognized. Supported metrics: 'auc', 'fit_auc', 'aac', 'fit_aac'.")
    cleaned_dfs = {}
    for name, dataset in dfs.items():
        df = dataset.format(data_type='experiments', metrics=metric)
        column = 'dose_response_value'
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')
        cleaned_df = (
            df.replace([np.inf, -np.inf], np.nan)
            .infer_objects(copy=False)
            .dropna(subset=[column])
        )

        # cleaned_df = cleaned_df[cleaned_df[column].between(thres, 1-thres)]
        if (metric in ["auc", "fit_auc"]):
            cleaned_df = cleaned_df[cleaned_df[column] <= thres]
        elif (metric in ["aac", "fit_aac"]):
            cleaned_df = cleaned_df[cleaned_df[column] >= thres]
        else:
            raise ValueError(f"Metric '{metric}' not recognized. Supported metrics: 'auc', 'fit_auc', 'aac', 'fit_aac'.")
        if cleaned_df[column].empty:
            print(f"⚠️ Column '{column}' in dataset is empty or non-numeric after cleaning.")
        cleaned_dfs[name] = cleaned_df
        print(f"Dataset '{name}': Cleaned {len(df) - len(cleaned_df)} rows with NaN or infinite '{column}' values.")
    return cleaned_dfs


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


def unify_cell_ids_across_dfs(dfs: dict):
    '''
    Get all cell ids from DataFrames return by the Dataset feature
    Args:
        dfs (dict): A dictionary where keys are dataset names and values are Dataset.
    
    Returns:
        dict: A dictionary where keys are dataset names and values are sets of cell ids.
    '''
    samples_in_datasets = {}
    for _, dataset in dfs.items():
        for name, data in dataset.items():
            if name not in samples_in_datasets:
                samples_in_datasets[name] = set()
            samples_in_datasets[name].update(data.index.tolist())
    
    for feature, dataset in dfs.items():
        for name, data in dataset.items():
            all_samples = list(samples_in_datasets[name])
            missing_samples = list(set(all_samples) - set(data.index.tolist()))
            if (len(missing_samples) > 0):
                print(f"Feature {feature}, Dataset {name}: Adding {len(missing_samples)} missing samples as part of total samples {len(all_samples)}.")
                if (feature == "proteomics"):
                    filling_df = pd.DataFrame(-30., index=missing_samples, columns=data.columns)
                elif (feature == "copy_number"):
                    filling_df = pd.DataFrame(2, index=missing_samples, columns=data.columns)
                else:
                    filling_df = pd.DataFrame(0, index=missing_samples, columns=data.columns)
                if (feature == "mutations"):
                    filling_df = pd.DataFrame(0, index=missing_samples, columns=data.columns)
                dataset[name] = pd.concat([data, filling_df])
            dataset[name] = dataset[name].reindex(index=all_samples)
        dfs[feature] = dataset
    return dfs


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
            df = np.log1p(df)
            all_columns.update(df.columns)
            sample_feature_dfs[name] = df
        df_dict_aligned = {
            name: df.reindex(columns=all_columns).fillna(0)
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
            name: df.reindex(columns=all_columns).fillna(0)
            for name, df in sample_feature_dfs.items()
        }
    elif (feature == "proteomics"):
        sample_feature_dfs = {}
        for name, dataset in dfs.items():
            df = dataset.format(data_type="proteomics")
            all_columns.update(df.columns)
            sample_feature_dfs[name] = df
        df_dict_aligned = {
            name: df.reindex(columns=all_columns).fillna(-30.)
            for name, df in sample_feature_dfs.items()
        }
    elif (feature == "copy_number"):
        sample_feature_dfs = {}
        for name, dataset in dfs.items():
            df = dataset.format(data_type="copy_number")
            all_columns.update(df.columns)
            sample_feature_dfs[name] = df
        df_dict_aligned = {
            name: df.reindex(columns=all_columns).fillna(2)
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


def binarize(df, nbit, shift=0.0):
    """
    Apply binarization to a DataFrame.

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
    if shift + min_value < 0:
        raise ValueError("Shift value is too small, resulting in negative values after shifting.")
    df = df + shift
    max_allowed = 2**nbit
    if (max_value + shift) > max_allowed:
        print(f"Warning: max value {max_value} after shifting exceeds {max_allowed}. Clipping to {max_allowed}.")
    df_bdd = df.clip(upper=max_allowed)
    # Perform binarization
    scaled_df = ((df_bdd / max_allowed) * (2**nbit - 1)).round().astype(int)
    # df_bits = scaled_df.applymap(lambda x: [int(b) for b in format(x, f'0{nbit}b')])
    # df_bits_array = df_bits.applymap(np.array)
    df_bits_array = scaled_df.apply(lambda col: col.map(lambda x: np.array([int(b) for b in format(x, f'0{nbit}b')], dtype=np.uint8)))
    return df_bits_array


def binarize_across_dfs(df_dict, feature: str):
    """
    Apply binarization across multiple DataFrames.

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
            df = df.replace([np.inf, -np.inf], 0).fillna(0)
            print(f"Binarizing transcriptomics data for {name} with shape {df.shape}")
            df_dict[name] = binarize(df, nbit=6, shift=0.0)
    elif (feature == "proteomics"):
        for name, df in df_dict.items():
            df = df.replace([np.inf, -np.inf], 0).fillna(-30.)
            print(f"Binarizing proteomics data for {name} with shape {df.shape}")
            df_dict[name] = binarize(df, nbit=6, shift=40.)
    elif (feature == "copy_number"):
        for name, df in df_dict.items():
            df = df.replace([np.inf, -np.inf], 2).fillna(2)
            print(f"Binarizing copy number data for {name} with shape {df.shape}")
            df_dict[name] = binarize(df, nbit=10, shift=0.0)
    else:
        print(f"Feature {feature} not supported for binarization")
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
            print(f"Feature {feature} is filtered out due to insufficient samples, more than {proportion*100}% samples required in at least 50% datasets.")

    return filtered_dfs


def plot_feature_variance_distribution(df_dict: dict, log_scale: bool = True, bins: int = 50):
    """
    Plot the distribution of feature variances for each dataset and overall,
    with proper log10 binning if log_scale=True, and filtering out non-positive values.

    Parameters
    ----------
    df_dict : dict
        A dictionary where keys are feature group names and values are dictionaries
        mapping dataset names to DataFrames.
    log_scale : bool, optional
        Whether to use log10 of variance for histogram binning.
    bins : int, optional
        Number of bins to use for the histogram.
    """
    for feature_group, datasets in df_dict.items():
        print(f"📊 Plotting variance distribution for feature group: {feature_group}")
        
        overall_var_list = []
        dataset_names = sorted(datasets.keys())

        for dataset_name in dataset_names:
            df = datasets[dataset_name]
            vars_ = df.var(axis=0).dropna()

            # Filter out non-positive variances (avoid log10(0) or log10(neg))
            positive_vars = vars_[vars_ > 0]
            n_removed = len(vars_) - len(positive_vars)
            if n_removed > 0:
                print(f"   ⚠️ {dataset_name}: Removed {n_removed} non-positive variances before log10 transform")

            if positive_vars.empty:
                print(f"   ⚠️ Skipping {feature_group} - {dataset_name}: no positive variances to plot")
                continue

            overall_var_list.append(positive_vars)

            plot_vals = np.log10(positive_vars) if log_scale else positive_vars

            plt.figure(figsize=(6, 4))
            plt.hist(plot_vals, bins=bins, edgecolor='black')
            xlabel = 'log10(Variance)' if log_scale else 'Variance'
            plt.xlabel(xlabel)
            plt.ylabel('Number of Features')
            plt.title(f'{feature_group} - {dataset_name}')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            # ✅ Add percentile markers
            percentiles = [30, 60, 90]
            percentile_vals = np.percentile(plot_vals, percentiles)
            for p, val in zip(percentiles, percentile_vals):
                plt.axvline(val, color='red', linestyle='--')
                plt.text(val, plt.ylim()[1]*0.9, f'{p}%', rotation=90,
                        verticalalignment='top', horizontalalignment='right', color='red', fontsize=9)
            plt.tight_layout()
            plt.show()


        # === Overall distribution ===
        if overall_var_list:
            overall_vars = pd.concat(overall_var_list, axis=1).max(axis=1).dropna()
            overall_vars = overall_vars[overall_vars > 0]  # filter again

            if not overall_vars.empty:
                plot_vals = np.log10(overall_vars) if log_scale else overall_vars

                plt.figure(figsize=(6, 4))
                plt.hist(plot_vals, bins=bins, edgecolor='black')
                xlabel = 'log10(Variance)' if log_scale else 'Variance'
                plt.xlabel(xlabel)
                plt.ylabel('Number of Features')
                plt.title(f'{feature_group} - overall (max across datasets)')
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                # ✅ Add percentile markers
                percentiles = [30, 60, 90]
                percentile_vals = np.percentile(plot_vals, percentiles)
                for p, val in zip(percentiles, percentile_vals):
                    plt.axvline(val, color='red', linestyle='--')
                    plt.text(val, plt.ylim()[1]*0.9, f'{p}%', rotation=90,
                            verticalalignment='top', horizontalalignment='right', color='red', fontsize=9)
                plt.tight_layout()
                plt.show()
            else:
                print(f"⚠️ Skipping {feature_group} - overall: no positive variances to plot")
        else:
            print(f"⚠️ Skipping {feature_group} - overall: no datasets with positive variances")


def count_features_by_variance_range(df_dict: dict, ranges: list):
    """
    Count the number of features whose variances fall into each specified range,
    including below_min and above_max bins.

    Parameters
    ----------
    df_dict : dict
        A dictionary where keys are feature group names and values are dictionaries
        mapping dataset names to DataFrames.
    ranges : list of tuple
        List of (low, high) tuples specifying variance ranges, sorted in ascending order.

    Returns
    -------
    dict
        A nested dictionary with structure:
        {
            feature_group: {
                dataset_name: {
                    "below_min": count,
                    "range_(low,high)": count,
                    ...,
                    "above_max": count
                },
                "overall": {...}
            }
        }
    """
    if not ranges:
        raise ValueError("ranges list must not be empty")

    counts = {}
    min_low = min(low for low, _ in ranges)
    max_high = max(high for _, high in ranges)

    for feature_group, datasets in df_dict.items():
        group_counts = {}
        overall_var_list = []

        for dataset_name, df in datasets.items():
            vars_ = df.var(axis=0)
            overall_var_list.append(vars_)
            dataset_counts = {}

            # Below min
            dataset_counts["below_min"] = int((vars_ < min_low).sum())

            # Each specified range
            for low, high in ranges:
                count_in_range = ((vars_ >= low) & (vars_ < high)).sum()
                dataset_counts[f"range_({low},{high})"] = int(count_in_range)

            # Above max
            dataset_counts["above_max"] = int((vars_ >= max_high).sum())

            group_counts[dataset_name] = dataset_counts

        # Overall (across datasets — max variance per feature)
        overall_vars = pd.concat(overall_var_list, axis=1).max(axis=1)
        overall_counts = {}

        overall_counts["below_min"] = int((overall_vars < min_low).sum())
        for low, high in ranges:
            count_in_range = ((overall_vars >= low) & (overall_vars < high)).sum()
            overall_counts[f"range_({low},{high})"] = int(count_in_range)
        overall_counts["above_max"] = int((overall_vars >= max_high).sum())

        group_counts["overall"] = overall_counts
        counts[feature_group] = group_counts

    return counts


def filter_feature(df_dict: dict, log_thres: float=-1.0):
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
        relative_log_vars = np.log10(0.05 * max_var_per_col.median() + 1e-8)
        print(f"relative_log_vars for {feature}: {relative_log_vars}, user defined log_thres: {log_thres}")
        selected_cols = max_var_per_col[np.log10(max_var_per_col + 1e-8) > log_thres].index.tolist()
        print(f"Feature {feature}: selected {len(selected_cols)} out of {vars_df.shape[0]} features")
        if len(selected_cols) == 0:
            print(f"Skipped: No features selected for {feature} after filtering with threshold {log_thres}.")
        else:
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
            # keep the distribution of response values
            value["response_bin"] = pd.qcut(value["dose_response_value"], q=5)
            sampled_df = value.groupby("response_bin", group_keys=False).apply(
                lambda x: x.sample(frac=frac, random_state=42)
            )
            df_dict_dipped[feature][dataset] = sampled_df.drop(columns=["response_bin"])
            # df_dict_dipped[feature][dataset] = value.sample(frac=frac, random_state=42)
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


def slice_data_across_dfs(df_dict, **kwargs):
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
    chunk_idx = kwargs.get("chunk_idx", 0)
    total_chunks = kwargs.get("total_chunks", 10)

    df_dict_sliced = {}
    for name, value in df_dict.items():
        total_size = value.shape[0]
        if total_size == 0:
            print(f"Warning: DataFrame '{name}' is empty. Skipping slicing.")
            continue
        chunk_size = total_size // total_chunks
        start_idx = chunk_idx * chunk_size
        end_idx = (chunk_idx + 1) * chunk_size if chunk_idx < total_chunks - 1 else total_size
        if start_idx >= end_idx:
            print(f"Warning: DataFrame '{name}' is too small to slice. Skipping.")
            continue
        df_dict_sliced[name] = value.iloc[start_idx:end_idx]
    return df_dict_sliced


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
        if isinstance(feature, numbers.Number):
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


def create_binary_cell_dfs(cell_feature_dfs):
    binary_cell_dfs = {}
    gene_mappings = {}
    for feature, datasets in cell_feature_dfs.items():
        for name, data in datasets.items():
            sample_ids = data.index.tolist()
            gene_map = data.columns.tolist()
            feature_encoding = np.stack([
                np.concatenate([
                    np.ravel(row[col]) if isinstance(row[col], np.ndarray) else np.array([row[col]])
                    for col in data.columns
                ])
                for _, row in data.iterrows()
            ])
            print(f"Dataset {name}, Feature {feature}: shape {feature_encoding.shape}")

            if name not in binary_cell_dfs:
                binary_cell_dfs[name] = pd.DataFrame(
                    index=sample_ids
                )
                binary_cell_dfs[name]["improve_sample_id"] = sample_ids
            binary_cell_dfs[name][feature] = pd.Series(list(feature_encoding), index=sample_ids)

            if feature not in gene_mappings:
                gene_mappings[feature] = gene_map
            else:
                if (gene_mappings[feature] != gene_map):
                    raise ValueError(f"Gene mapping mismatch for dataset {name} and feature {feature}, should have unified mapping across datasets.") 

    return binary_cell_dfs, gene_mappings


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
        if show_status:
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
        if show_status:
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


def cat_drug_response_features_across_datasets(features_dfs:dict, labels_dfs: dict, cell_id_ref_dfs: dict, feature_list: list, show_status:bool=False) -> np.ndarray:
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
        if show_status:
            print(f"Dataset '{name}': {data.shape}")
        if data.size == 0 or label.size == 0:
            if show_status:
                print(f"Warning: Dataset '{name}' has no valid samples after feature extraction. Skipping.")
            continue
        training_datas.append(data)
        training_labels.append(label)
    return np.concatenate(training_datas, axis=0), np.concatenate(training_labels, axis=0)


def cat_cell_features_across_datasets(features_dfs:dict, feature_list: list=None, show_status:bool=False) -> np.ndarray:
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
    if feature_list is None:
        feature_list = set()
        for name, data in features_dfs.items():
            feature_list.update(data.columns.tolist())
        feature_list = list(feature_list)
        feature_list.remove("improve_sample_id")
    for name, data in features_dfs.items():
        X = np.hstack([
            np.vstack(data[col].to_numpy()) if isinstance(data[col].iloc[0], np.ndarray)
            else data[[col]].to_numpy()
            for col in feature_list
        ])
        training_datas.append(X)
        training_labels.append(data["improve_sample_id"].to_numpy().reshape(X.shape[0], 1))
        if show_status:
            print(f"Dataset '{name}': {data.shape}")
    return np.concatenate(training_datas, axis=0), np.concatenate(training_labels, axis=0)


def cat_drug_response_features(features:dict, labels: list, cell_id_ref: dict, feature_list: list, show_status:bool=False) -> np.ndarray:
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
                if isinstance(features[feature][s_id], np.ndarray):
                    cell_features.extend(features[feature][s_id])
                else:
                    cell_features.append(features[feature][s_id])
            else:
                raise KeyError(f"Feature '{feature}' not found in features dictionary.")
        cell_features.append(features["fingerprint"][s_id])
        training_data.append(np.concatenate(cell_features, axis=0))
    training_data = np.array(training_data)
    training_label = np.array(training_label)
    return training_data, training_label


def cat_cell_features(features:dict, feature_list: list=None, show_status:bool=False) -> np.ndarray:
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


def save_dfs_as_pt(
    experiments_dfs,
    drugs_ref_dfs,
    dim_reduced_dfs,
    unique_cell_ids_across_dfs,
    features=["transcriptomics", "proteomics", "copy_number"],
    n_chunks=100,
    save_dir=f"../data/pt_data/",
    show_status:bool=False
):
    """
    Convert large experiment DataFrames into PyTorch tensors and save them as .pt files in chunks.
    
    Parameters
    ----------
    experiments_dfs : list of pd.DataFrame
        List of experiment dataframes.
    drugs_ref_dfs : list of pd.DataFrame
        Reference dataframes for drugs.
    dim_reduced_dfs : list of pd.DataFrame
        Dimensionality reduced datasets.
    unique_cell_ids_across_dfs : list
        Unique cell IDs across all datasets.
    features : list of str
        List of feature types to extract.
    fraction : float
        Fraction of data to sample at a time.
    chunk_size : int
        Number of experiments to process per chunk.
    save_dir : str
        Directory to save PyTorch tensor files.
    """
    save_dir = save_dir + "_".join(features) + "/"
    os.makedirs(save_dir, exist_ok=True)
        
    start_idx = 0
    for chunk_id in tqdm(range(n_chunks), desc="Saving chunks"):
        
        # Sample a fraction of the data
        experiments_dfs_sliced = slice_data_across_dfs(
            experiments_dfs, 
            chunk_idx=chunk_id,
            total_chunks=n_chunks
        )

        # Preprocess to get feature dict and response
        feature_dict_footprint, resp_footprint = preprocess_experiment_with_footprint(
            experiments_dfs_sliced, 
            drugs_ref_dfs, 
            dim_reduced_dfs, 
            drug_target=None,
            show_status=show_status
        )
        
        # Concatenate features across datasets
        X_chunk_np, y_chunk_np = cat_drug_response_features_across_datasets(
            feature_dict_footprint, 
            resp_footprint, 
            unique_cell_ids_across_dfs, 
            features,
            show_status=show_status
        )

        if X_chunk_np is None or getattr(X_chunk_np, "size", 0) == 0:
            print(f"  Chunk {chunk_id} produced empty output -> skipped")
            continue

        # Convert to PyTorch tensors
        X_chunk = torch.from_numpy(X_chunk_np).float()
        y_chunk = torch.from_numpy(y_chunk_np).float()

        del X_chunk_np, y_chunk_np  # Free up memory
        
        # Save chunk as .pt
        torch.save((X_chunk, y_chunk), os.path.join(save_dir, f"chunk_{chunk_id}.pt"))
        
        start_idx += X_chunk.shape[0]
        
        del X_chunk, y_chunk  # Free up memory
        

    print(f"Saved {n_chunks} chunks to {save_dir}.")
    print("You can later load them with torch.load for PyTorch usage.")


