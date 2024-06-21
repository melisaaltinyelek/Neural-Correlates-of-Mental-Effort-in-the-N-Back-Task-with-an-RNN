#%%

import tensorflow as tf
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval

# %%

def preprocess_data(data_as_input):
    
    # Read the dataset
    df = pd.read_csv(data_as_input)

    # Convert the response column into one-hot encoded labels
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
    dataframe = df.reindex(columns = ["encoded_letter", "one_hot_response"])

    return dataframe

dataset = preprocess_data("data.csv")


# %%

# def train_model(train_df, n_neurons = None, n_batch = None, learning_rate = None):

#     # X_train = [literal_eval(x) for x in train_df["letter"].tolist()]
#     # y_train = [literal_eval(x) for x in train_df["one_hot_response"].tolist()]

#     # X_train = np.array(X_train)
#     # y_train = np.array(y_train)

    
#     pass

# X_train, y_train = train_model(train_df = train_df)
# %%
