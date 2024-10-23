#%%

import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
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

        # Convert the encoded columns to lists to store them in pandas DataFrame
        df["letter"] = encoded_letters.tolist()
        df["response"] = encoded_responses.tolist()

        self.df = df

        print(self.df)
        # df.to_csv("nback_data.csv", index = False)

        return df
  
    def create_sequences(self, df, n_steps = 4):

        X, y = [], []

        letters = np.array(df["letter"].tolist())
        responses = np.array(df["response"].tolist())

        
        # Create chunks or sequences of 4 consecutive letters, store them in X
        # For each sequence in X, store the response for the last letter in that sequence
        for i in range(len(letters) - n_steps):
            X.append(letters[i:i + n_steps])
            #print(X)
            y.append(responses[i + n_steps - 1])
            #print(y)

        # Reshape the X (letters) as (num_samples, 1, num_features) so that LSTM can receive 1 letter at a time
        X = np.array(X)
        y = np.array(y)

        # print(f"The shape of the letters: {letters.shape}") # 6 --> the number of features/letters
        # print("-" * 50)
        # print(f"The shape of the responses: {responses.shape}") 
        print(f"This is the X (letters): {X}")
        print("-" * 50)
        print(f"This is the length of X: {len(X)}")
        print("-" * 50)
        print(f"This is the y (responses): {y}")
        print("-" * 50)
        print(f"This is the length of y: {len(y)}")
        print("-" * 50)

        return X, y

    def split_data(self, data, train_ratio = 0.8):

        X, y = self.create_sequences(data)
        
        total_samples = len(X)
        # print(total_samples)
        train_size = int(train_ratio * total_samples)
        # print(train_size)
        
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
            tf.keras.layers.LSTM(6, input_shape = (1, self.X_train.shape[2])),
            # tf.keras.layers.Dropout(0.2),  
            tf.keras.layers.Dense(12, activation = "relu"),
            tf.keras.layers.Dense(3, activation = "softmax")
        ])
        
        self.model.compile(optimizer = tf.keras.optimizers.Adam(self.learning_rate),
                           loss = "categorical_crossentropy",
                           metrics = ["accuracy"])

    def train_model(self, epochs):

        cce_history = []    

        class_weights_dict = {
            0: 0.5, # Lure
            1: 1.0, # Nontarget
            2: 1.0 # Target
        }

        self.training_history = self.model.fit(self.X_train, self.y_train,
                                          epochs = epochs,
                                          batch_size = self.n_batch,
                                          validation_data = (self.X_val, self.y_val),
                                          class_weight = class_weights_dict,
                                          shuffle = True)
                                        
        cce_history.append(self.training_history.history)

        # print(training_history.history.keys())

        return cce_history, self.model, self.training_history
    
    def eval_model(self):

        eval_results = self.model.evaluate(self.X_test, self.y_test, batch_size = 128)
        print(f"Overall Test loss, Test accuracy: {eval_results}")

        predictions = self.model.predict(self.X_test)

        # Print the first 10 predictions along with their true values
        print("\nPredictions for 10 trials:")
        for i in range(10):
            print(f"Trial {i + 1}:")
            print(f"Predicted: {predictions[i]}, True value: {self.y_test[i]}")

        return eval_results
        
    def visualize_results(self):
        
        # Plot history for accuracies
        plt.plot(self.training_history.history["accuracy"], color = "purple")
        plt.plot(self.training_history.history["val_accuracy"], color = "green")
        plt.title("Model Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Test"], loc = "upper right")
        plt.show()

        # Plot history for losses
        plt.plot(self.training_history.history["loss"], color = "purple")
        plt.plot(training_history.history["val_loss"], color = "green")
        plt.title("Model Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Test"], loc = "upper right")
        plt.show()

#%%

if __name__ == "__main__":

    data_processor = DataPreprocessor("raw_data.csv")  
    preprocessed_data = data_processor.preprocess_data()

    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.split_data(preprocessed_data)

    print("Training dataset shape:", X_train.shape, y_train.shape)
    print("Validation dataset shape:", X_val.shape, y_val.shape)
    print("Test dataset shape:", X_test.shape, y_test.shape)

    lstm_trainer = LSTMTrainer(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, n_batch = 64, learning_rate = 0.01)
    lstm_trainer.initialize_model()
    cce_history, model, training_history = lstm_trainer.train_model(epochs = 100)
    eval_results = lstm_trainer.eval_model()
    display_acc_loss = lstm_trainer.visualize_results()

# %%
