#%%

import tensorflow as tf
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# %%

def split_dataframe(data_as_input):
    
    df = pd.read_csv(data_as_input)

    df["one_hot_response"] = df["response"].apply(lambda x: 1 if x == "target" else 0)
    
    midpoint = len(df) // 2
    
    train_df = df.iloc[:midpoint]
    val_df = df.iloc[midpoint:]
    
    return train_df, val_df

train_df, val_df = split_dataframe("data.csv")

train_df.to_csv("train_data.csv", index = False)
val_df.to_csv("val_data.csv", index = False)

# %%

