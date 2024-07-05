#%%

import tensorflow as tf
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval
import tensorflow_datasets as tfds
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# %%

def preprocess_data(data_as_input):
    
    # Read the dataset
    df = pd.read_csv(data_as_input)

    # Create a mapping for one-hot encoding
    encoded_letters = {
        "A": [1, 0, 0, 0, 0, 0],
        "B": [0, 1, 0, 0, 0, 0],
        "C": [0, 0, 1, 0, 0, 0],
        "D": [0, 0, 0, 1, 0, 0],
        "E": [0, 0, 0, 0, 1, 0],
        "F": [0, 0, 0, 0, 0, 1]
    }

    df["letter"] = df["letter"].map(encoded_letters)

    encoded_responses = {
        "target": [1, 0, 0],
        "no target": [0, 1, 0],
        "is lure": [0, 0, 1]
    }
    
    df["response"] = df["response"].map(encoded_responses)
    
    #df = df.dropna().reset_index(drop=True)
    return df

dataset = preprocess_data("data.csv")
#dataset.to_csv("new_data.csv", index = False)

#%%

def split_data(data_to_split):

    df = pd.read_csv(data_to_split)

    letters = np.array(df["letter"].tolist())
    responses = np.array(df["response"].tolist())

    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    test_size = len(df) - train_size - val_size

    X_train = letters[:train_size]
    y_train = responses[:train_size]

    X_val = letters[train_size:val_size + train_size]
    y_val = responses[train_size:val_size + train_size]

    X_test = letters[train_size + val_size:]
    y_test = responses[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data("n_back_data.csv")

print("Training dataset shape:", X_train.shape, y_train.shape)
print("Validation dataset shape:", X_val.shape, y_val.shape)
print("Test dataset shape:", X_test.shape, y_test.shape)

#%%

# Assign training, validation and test samples as 80%, 10% and 10%, respectively
# train_size = int(0.8 * len(dataframe))
# val_size = int(0.1 * len(dataframe))
# test_size = len(dataframe) - train_size - val_size

# # Create a copy of the original dataset to avoid modifications
# df_train_val = dataframe.copy() 

# # Extract validation set first
# df_val = df_train_val.iloc[:val_size]

# # Extract training set using calculated size
# df_train = df_train_val.iloc[val_size:val_size+train_size]

# # Testing set remains separate (order preserved from original dataset)
# df_test = df_train_val.iloc[val_size+train_size:]

#%%

