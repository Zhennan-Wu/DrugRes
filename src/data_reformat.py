import coderdata as cd
import os
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # Correct import for new RDKit
import pubchempy as pcp

import coderdata_processing
import importlib
importlib.reload(coderdata_processing)
from coderdata_processing import filter_biomarkers, filter_feature, unify_feature_across_dataset, get_total_mutation_types, summarize_datasets, summarize_df_matrices, binarize_across_dfs,  get_cell_id_across_dfs, clean_experiments_across_dfs, save_dfs_as_pt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def clean_data(download=False, show_status=False):
    datasets = ["beataml", "bladder", "ccle", "colorectal", "ctrpv2", "fimm", "gcsi", "gdscv1", "gdscv2", "liver", "mpnst", "nci60", "pancreatic", "prism"]

    include_copy_number = True
    include_proteomics = True

    to_remove = ["cptac", "hcmi"]
    if (include_copy_number):
        new_to_remove = ["beataml", "sarcoma"]
        to_remove = list(set(to_remove + new_to_remove))

    if (include_proteomics):
        new_to_remove = ["bladder", "colorectal", "novartis", "pancreatic", "sarcoma"]
        to_remove = list(set(to_remove + new_to_remove))

    datasets = [x for x in datasets if x not in to_remove]
   
    data_path = '../data/Cancer/coderdata/'
    if download:
        for data_name in datasets:
            cd.download(name=data_name, local_path=data_path)

    selected_datasets = {}
    for data_name in datasets:
        if show_status:
            print(f"Loading dataset: {data_name}")
        selected_datasets[data_name] = cd.load(name=data_name, local_path=data_path)

    unified_datasets = {}
    unified_datasets["transcriptomics"] = unify_feature_across_dataset(selected_datasets, feature="transcriptomics")
    if include_proteomics:
        unified_datasets["proteomics"] = unify_feature_across_dataset(selected_datasets, feature="proteomics")
    if include_copy_number:
        unified_datasets["copy_number"] = unify_feature_across_dataset(selected_datasets, feature="copy_number")
    # for mutations, need to get total mutation variation types
    all_mutation_types = get_total_mutation_types(selected_datasets)
    for mut_type in all_mutation_types:
        unified_datasets[f"mutations_{mut_type}"] = unify_feature_across_dataset(selected_datasets, feature="mutations", mutation_type=mut_type)
    unique_cell_ids_across_dfs = get_cell_id_across_dfs(unified_datasets)
    if show_status:
        print(f"Unified features {unified_datasets.keys()} across datasets.")

    data_size_ref = {}
    for data, value in unified_datasets["transcriptomics"].items():
        data_size_ref[data] = value.shape[0]

    feature_reduced_dfs = filter_biomarkers(unified_datasets, data_size_ref, proportion=1e-2)

    dim_reduced_dfs = filter_feature(feature_reduced_dfs)
    # Normalization
    dim_reduced_dfs["transcriptomics"] = binarize_across_dfs(dim_reduced_dfs["transcriptomics"], "transcriptomics")
    if include_proteomics:
        dim_reduced_dfs["proteomics"] = binarize_across_dfs(dim_reduced_dfs["proteomics"], "proteomics")
    if include_copy_number:
        dim_reduced_dfs["copy_number"] = binarize_across_dfs(dim_reduced_dfs["copy_number"], "copy_number")

    drugs_ref_dfs = {}
    for name, data in selected_datasets.items():
        drugs_ref_dfs[name] = data.drugs.drop_duplicates(subset=["improve_drug_id"], inplace=False)

    metric = "fit_auc"
    experiments_dfs = clean_experiments_across_dfs(selected_datasets, metric=metric)
    print("Data ready for reformat and store")
    return experiments_dfs, drugs_ref_dfs, dim_reduced_dfs, unique_cell_ids_across_dfs


if __name__ == "__main__":
    experiments_dfs, drugs_ref_dfs, dim_reduced_dfs, unique_cell_ids_across_dfs = clean_data(download=True)

    save_dfs_as_pt(experiments_dfs,
    drugs_ref_dfs,
    dim_reduced_dfs,
    unique_cell_ids_across_dfs,
    features=["transcriptomics", "proteomics", "copy_number"],
    n_chunks=100,
    save_dir=f"../data/pt_data/"
    )