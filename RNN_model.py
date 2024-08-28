#%%
import tensorflow as tf
import tensorflow_datasets as tfds
#from tensorflow.keras.regularizers import l2 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval
import pandas as pd
import numpy as np
import glob

# %%

class DataPreprocessor:
    def __init__(self, data_path):

        self.data_path = data_path
        self.df = None

    def preprocess_data(self):

        df = pd.read_csv(self.data_path)

        letter_encoder = OneHotEncoder(sparse_output = False) # sparse_output = False returns a NumPy array
        response_encoder = OneHotEncoder(sparse_output = False) # sparse_output = False returns a NumPy array

        encoded_letters = letter_encoder.fit_transform(df[["letter"]])
        encoded_responses = response_encoder.fit_transform(df[["response"]])

        df["letter"] = encoded_letters.tolist()
        df["response"] = encoded_responses.tolist()

        self.df = df

        print(self.df)
        #df.to_csv("nback_data.csv", index = False)

        return df
  
    def create_sequences(self, df, n_steps = 4):

        letters = np.array(df["letter"].tolist())
        responses = np.array(df["response"].tolist())

        X, y = [], []

        for i in range(len(letters) - n_steps):
            X.append(letters[i:i + n_steps])
            #print(X)
            y.append(responses[i + n_steps - 1])
            #print(y)

        return np.array(X), np.array(y)

    def split_data(self, data, train_ratio = 0.8, val_ratio = 0.2):

        X, y = self.create_sequences(data)

        total_samples = len(X)
        train_size = int(train_ratio * total_samples)
        
        remaining_samples = total_samples - train_size
        # Divide val_size by 2 so that both validation and test datasets have the same number of data
        val_size = remaining_samples // 2 
        # test_size = remaining_samples - val_size

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

class LSTMTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, n_batch, learning_rate):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.n_batch = n_batch
        self.learning_rate = learning_rate
        self.model = None

    def initialize_model(self):

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(12, input_shape = (self.X_train.shape[1], self.X_train.shape[2])),
            tf.keras.layers.Dropout(0.2),  
            tf.keras.layers.Dense(32, activation = "relu"),
            tf.keras.layers.Dense(3, activation = "softmax")
        ])
        
        self.model.compile(optimizer = tf.keras.optimizers.Adam(self.learning_rate),
                           loss = "categorical_crossentropy",
                           metrics = ["accuracy"])

    def train_model(self, epochs):

        cce_history = []

        training_history = self.model.fit(self.X_train, self.y_train,
                                          epochs = epochs,
                                          batch_size = self.n_batch,
                                          validation_data = (self.X_val, self.y_val),
                                          shuffle = True)
                                        

        cce_history.append(training_history.history)

        return cce_history, self.model
    
    def eval_model(self):

        eval_results = self.model.evaluate(self.X_test, self.y_test, batch_size = 128)
        print(f"Evaluation result:{eval_results}")

        return eval_results

#%%

if __name__ == "__main__":

    data_processor = DataPreprocessor("raw_data.csv")  
    preprocessed_data = data_processor.preprocess_data()

    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.split_data(preprocessed_data)

    print("Training dataset shape:", X_train.shape, y_train.shape)
    print("Validation dataset shape:", X_val.shape, y_val.shape)
    print("Test dataset shape:", X_test.shape, y_test.shape)

    lstm_trainer = LSTMTrainer(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, n_batch = 128, learning_rate = 0.01)
    lstm_trainer.initialize_model()
    history, model = lstm_trainer.train_model(epochs = 200)
    eval_results = lstm_trainer.eval_model()

# %%
