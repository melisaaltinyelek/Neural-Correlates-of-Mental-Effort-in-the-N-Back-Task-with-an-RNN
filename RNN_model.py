#%%

import tensorflow as tf
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval
import tensorflow_datasets as tfds

# %%

def preprocess_data(data_as_input):
    
    # Read the dataset
    df = pd.read_csv(data_as_input)

    # Create a new column "is_lure" and transfer the values that represent lure trials from the "response" column
    df["is_lure"] = df["response"].apply(lambda x: x if x == "is lure" else None)

    # Convert the "is_lure" column into one-hot encoded lanels
    df["is_lure"] = df["is_lure"].apply(lambda x: 1 if x == "is lure" else 0)

    # Convert the "response" column into one-hot encoded labels
    df["one_hot_response"] = df["response"].apply(lambda x: 1 if x == "target" else 0)

    # Sort the unique letters in the train_df
    letters_sorted = sorted(df['letter'].unique())

    # Iterate over the letters and map each of them to a number
    # {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    letter_to_int = {letter: i for i, letter in enumerate(letters_sorted)}
    df['encoded_letter'] = df['letter'].map(letter_to_int)

    # # Drop the old "letter" column
    df.drop(columns = "letter")

    # Remove the old "response" column
    df.drop(columns = ["response"])

    # # Reorder the columns of train_df
    dataframe = df.reindex(columns = ["encoded_letter", "one_hot_response", "is_lure"])

    return dataframe

dataset = preprocess_data("data.csv")
dataset.to_csv("new_data.csv", index = True)

#%%

# Assign training, validation and test sizes as 80%, 10% and 10%, respectively
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
#test_size = int(0.1 * len(dataset))

# Create a copy of the original dataset to avoid modifications
df_train_val = dataset.copy() 

# Extract validation set first
df_val = df_train_val.iloc[:val_size]

# Extract training set using calculated size
df_train = df_train_val.iloc[val_size:val_size+train_size]

# Testing set remains separate (order preserved from original dataset)
df_test = df_train_val.iloc[val_size+train_size:]

#%%


