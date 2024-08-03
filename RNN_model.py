#%%

import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from ast import literal_eval
import pandas as pd
import glob
import numpy as np


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

        return df
  
    def create_sequences(self, data, n_steps = 4):

        letters = np.array(data["letter"].tolist())
        responses = np.array(data["response"].tolist())

        X, y = [], []
        for i in range(len(letters) - n_steps):
            X.append(letters[i:i + n_steps])
            y.append(responses[i + n_steps])

        return np.array(X), np.array(y)

    def split_data(self, data, train_ratio = 0.8, val_ratio = 0.1):

        X, y = self.create_sequences(data)

        train_size = int(train_ratio * len(X))
        val_size = int(val_ratio * len(X))
        #test_size = len(X) - train_size - val_size

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

class LSTMTrainer:
    def __init__(self, X_train, y_train, n_batch, learning_rate):

        self.X_train = X_train
        self.y_train = y_train
        self.n_batch = n_batch
        self.learning_rate = learning_rate
        self.model = None

    def initialize_model(self):

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape = (self.X_train.shape[1], self.X_train.shape[2]), return_sequences = True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(128, return_sequences = False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation = "softmax")
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss = "categorical_crossentropy",
                           metrics = ["accuracy"])

    def train_model(self, epochs):

        cce_history = []

        training_history = self.model.fit(self.X_train, self.y_train,
                                          epochs = epochs,
                                          batch_size = self.n_batch,
                                          #validation_split=0.2,
                                          shuffle = True)
                                          #callbacks=[early_stopping, reduce_lr])

        cce_history.append(training_history.history)
        return cce_history, self.model

#%%

if __name__ == "__main__":

    data_processor = DataPreprocessor("raw_data.csv")  
    preprocessed_data = data_processor.preprocess_data()

    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.split_data(preprocessed_data)

    print("Training dataset shape:", X_train.shape, y_train.shape)
    print("Validation dataset shape:", X_val.shape, y_val.shape)
    print("Test dataset shape:", X_test.shape, y_test.shape)

    lstm_trainer = LSTMTrainer(X_train = X_train, y_train = y_train, n_batch = 64, learning_rate = 0.01)
    lstm_trainer.initialize_model()
    history, model = lstm_trainer.train_model(epochs = 200)


