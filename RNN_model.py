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

class DataPreprocessor:
  
  def __init__(self, data_path):

    self.data_path = data_path

    self.encoded_letters = {
        "A": [1, 0, 0, 0, 0, 0],
        "B": [0, 1, 0, 0, 0, 0],
        "C": [0, 0, 1, 0, 0, 0],
        "D": [0, 0, 0, 1, 0, 0],
        "E": [0, 0, 0, 0, 1, 0],
        "F": [0, 0, 0, 0, 0, 1]
    }

    self.encoded_responses = {
        "target": [1, 0, 0],
        "lure": [0, 1, 0],
        "nontarget": [0, 0, 1]
    }

  def preprocess_data(self):
    
    df = pd.read_csv(self.data_path)

    df["letter"] = df["letter"].map(self.encoded_letters)
    df["response"] = df["response"].map(self.encoded_responses)

    #df.to_csv("nback_data.csv", index = False)

    return df

  def split_data(self, data, train_ratio = 0.8, val_ratio = 0.1):

    letters = np.array(data["letter"].tolist())
    responses = np.array(data["response"].tolist())

    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    test_size = len(data) - train_size - val_size

    X_train = letters[:train_size]
    y_train = responses[:train_size]

    X_val = letters[train_size:val_size + train_size]
    y_val = responses[train_size:val_size + train_size]

    X_test = letters[train_size + val_size:]
    y_test = responses[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test
  

class LSTMTrainer:
  def __init__(self, X_train, y_train, n_batch, learning_rate):

    # X_train.shape = (3762, 1, 6) (# samples/, 1 timestep, 6 features)
    self.X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    self.y_train = y_train
    self.n_batch = n_batch
    self.learning_rate = learning_rate
    self.model = None
  
    # X_train.shape = (3762, 1, 6) (# samples/, 1 timestep, 6 features)
    #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

  def initialize_model(self):

    self.model = tf.keras.Sequential([
      tf.keras.layers.LSTM(32, input_shape = (self.X_train.shape[1], self.X_train.shape[2])),
      tf.keras.layers.Dense(64, activation = 'relu'),
      tf.keras.layers.Dense(3, activation = "softmax")
    ])

    self.model.compile(optimizer = tf.keras.optimizers.Adam(self.learning_rate),
                    loss = tf.keras.losses.CategoricalCrossentropy(),
                    metrics = ["accuracy"])
    
  def train_model(self, epochs):

    bce_history = []

    training_history = self.model.fit(self.X_train, self.y_train,
                                 epochs = epochs,
                                 batch_size = self.n_batch,
                                 shuffle = True)
    
    bce_history.append(training_history.history)

    return bce_history, self.model 

#%%

if __name__ == "__main__":
  
  data_processor = DataPreprocessor("raw_data.csv")  
  preprocessed_data = data_processor.preprocess_data()

  X_train, y_train, X_val, y_val, X_test, y_test = data_processor.split_data(preprocessed_data)

  print("Training dataset shape:", X_train.shape, y_train.shape)
  print("Validation dataset shape:", X_val.shape, y_val.shape)
  print("Test dataset shape:", X_test.shape, y_test.shape)

  lstm_trainer = LSTMTrainer(X_train = X_train, y_train = y_train, n_batch = 32, learning_rate = 0.01)
  lstm_trainer.initialize_model()
  history = lstm_trainer.train_model(epochs = 100)


  # def train_model(X_train, y_train, n_batch, learning_rate):

  #   # X_train.shape = (3762, 1, 6) (# samples/, 1 timestep, 6 features)
  #   X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

  #   model = tf.keras.Sequential([
  #     tf.keras.layers.LSTM(32, input_shape = (X_train.shape[1], X_train.shape[2])),
  #     tf.keras.layers.Dense(2, activation = "sigmoid"),
  #   ])

  #   #optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

  #   model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate),
  #                   loss = tf.keras.losses.BinaryCrossentropy(),
  #                   metrics = ["accuracy"])
      

  #   bce_history = []

  #   training_history = model.fit(X_train, y_train,
  #                                epochs = 50,
  #                                batch_size = n_batch,
  #                                verbose = 1,
  #                                shuffle = False)


  #   bce_history.append(training_history.history)

  #   print(X_train.shape)

  #   return bce_history, model
  
  
  #bce_history, model = train_model(X_train = X_train, y_train = y_train, n_batch = 32, learning_rate = 0.1)

#%%

# def preprocess_data(data_as_input):
    
#     # Read the dataset
#     df = pd.read_csv(data_as_input)

#     # Create a mapping for one-hot encoding
#     encoded_letters = {
#         "A": [1, 0, 0, 0, 0, 0],
#         "B": [0, 1, 0, 0, 0, 0],
#         "C": [0, 0, 1, 0, 0, 0],
#         "D": [0, 0, 0, 1, 0, 0],
#         "E": [0, 0, 0, 0, 1, 0],
#         "F": [0, 0, 0, 0, 0, 1]
#     }

#     df["letter"] = df["letter"].map(encoded_letters)

#     encoded_responses = {
#         "target": [1, 0, 0],
#         "no target": [0, 1, 0],
#         "is lure": [0, 0, 1]
#     }
    
#     df["response"] = df["response"].map(encoded_responses)
    
#     #df = df.dropna().reset_index(drop=True)
#     return df

# dataset = preprocess_data("data.csv")
# #dataset.to_csv("new_data.csv", index = False)

# #%%

# def split_data(data_to_split):

#     df = pd.read_csv(data_to_split)

#     letters = np.array(df["letter"].tolist())
#     responses = np.array(df["response"].tolist())

#     train_size = int(0.8 * len(df))
#     val_size = int(0.1 * len(df))
#     test_size = len(df) - train_size - val_size

#     X_train = letters[:train_size]
#     y_train = responses[:train_size]

#     X_val = letters[train_size:val_size + train_size]
#     y_val = responses[train_size:val_size + train_size]

#     X_test = letters[train_size + val_size:]
#     y_test = responses[train_size + val_size:]

#     return X_train, y_train, X_val, y_val, X_test, y_test

# X_train, y_train, X_val, y_val, X_test, y_test = split_data("n_back_data.csv")

# print("Training dataset shape:", X_train.shape, y_train.shape)
# print("Validation dataset shape:", X_val.shape, y_val.shape)
# print("Test dataset shape:", X_test.shape, y_test.shape)

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

