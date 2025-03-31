import pandas as pd
import numpy as np
import coderdata as cd
from numpy import loadtxt
import matplotlib.pyplot as plt


def aggre_coderdata_drugres(dataset_name: str, features: list, response_metric: str, path: str ='../data/Cancer/coderdata/'):
    """
    Get the data from the coderdata package
    Assume all dataset has attributes:
    - 'transcriptomics',
    - 'proteomics',
    - 'mutations',
    - 'copy_number',
    - 'samples',
    - 'drugs',
    - 'experiments',
    - 'genes',
    """
    cd.download(name=dataset_name,  local_path=path)
    data = cd.load(name=dataset_name, local_path=path)

    merged_df = pd.merge(data.experiments[data.experiments["dose_response_metric"] == response_metric][['improve_sample_id', 'improve_drug_id', 'dose_response_value']], 
                         data.drugs[['improve_drug_id', 'canSMILES']],
                         on='improve_drug_id',
                         how='inner').drop_duplicates()
    merged_df = merged_df.dropna(subset=['canSMILES'])
    merged_df