
import pandas as pd


def min_max_scaling(df):
    min_value = df.min().min()
    max_value = df.max().max()
    # Perform min-max scaling
    scaled_df = (df - min_value) / (max_value - min_value)
    return scaled_df

