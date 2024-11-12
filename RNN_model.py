#%%

import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from ast import literal_eval
import pandas as pd
import numpy as np
import os
import glob

# %%

class DataPreprocessor:
    def __init__(self, data_path_bin_cl, data_path_multiclass_pred):

        self.data_path_bin_cl = data_path_bin_cl
        self.data_path_multiclass_pred = data_path_multiclass_pred
        self.df = None
        self.df_with_lures = None

    def preprocess_data_for_bin_cl(self):

        df = pd.read_csv(self.data_path_bin_cl)

        df.replace(to_replace = "lure", value = "nontarget", inplace = True)

        response_binary_mapping = {
            "nontarget": 0,
            "target": 1
        }
        
        df["response"] = df["response"].map(response_binary_mapping)

        letter_encoder = OneHotEncoder(sparse_output = False) # sparse_output = False returns a NumPy array
        # response_encoder = OneHotEncoder(sparse_output = False) # sparse_output = False returns a NumPy array

        encoded_letters = letter_encoder.fit_transform(df[["letter"]])
        # encoded_responses = response_encoder.fit_transform(df[["response"]])

        # # Convert the encoded columns to lists to store them in pandas DataFrame
        df["letter"] = encoded_letters.tolist()
        # df["response"] = encoded_responses.tolist()
        
        self.df = df

        # df.to_csv("3-back task/nback_data_without_lure.csv", index = False)

        # print(self.df)

        return self.df
    
    def preprocess_data_for_multic_pred(self):

        df_with_lures = pd.read_csv(self.data_path_multiclass_pred)

        response_multiclass_mapping = {
            "nontarget": 0,
            "target": 1,
            "lure": 2
        }

        df_with_lures["response"] = df_with_lures["response"].map(response_multiclass_mapping)

        test_letter_encoder = OneHotEncoder(sparse_output = False)
        test_encoded_letters = test_letter_encoder.fit_transform(df_with_lures[["letter"]])

        df_with_lures["letter"] = test_encoded_letters.tolist()
        
        self.df_with_lures = df_with_lures

        # df_with_lures.to_csv("3-back task/nback_data_with_lures.csv", index = False)

        return self.df_with_lures
        
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

    def split_data_for_bin_pred(self, data, train_ratio = 0.8):

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
    
    def split_data_for_multic_pred(self, data):

        X_test_with_lures, y_test_with_lures = self.create_sequences(data)

        print(f"This is X_test with lures: {X_test_with_lures}")
        print("-" * 50)
        print(f"This is y_test with lures: {y_test_with_lures}")
        print("-" * 50)
        print(f"This is the length of X_test_with_lures: {len(X_test_with_lures)}")
        print("-" * 50)
        print(f"This is the length of y_test_with_lures: {len(y_test_with_lures)}")
        print("-" * 50)

        return X_test_with_lures, y_test_with_lures

class LSTMTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, X_test_with_lures, y_test_with_lures, n_batch, learning_rate):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_with_lures = X_test_with_lures
        self.y_test_with_lures = y_test_with_lures
        self.n_batch = n_batch
        self.learning_rate = learning_rate
        self.model = None

    def initialize_model(self):

        self.model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(6, input_shape = (1, self.X_train.shape[2])),
            # tf.keras.layers.Dropout(0.2),  
            tf.keras.layers.Dense(12, activation = "relu"),
            tf.keras.layers.Dense(1, activation = "sigmoid")
        ])
        
        self.model.compile(optimizer = tf.keras.optimizers.Adam(self.learning_rate),
                           loss = "binary_crossentropy",
                           metrics = ["accuracy"])

    def train_model(self, epochs):

        bce_history = []    

        # class_weights_dict = {
        #     0: 0.5, # Lure
        #     1: 1.0, # Nontarget
        #     2: 1.0 # Target
        # }

        self.training_history = self.model.fit(self.X_train, self.y_train,
                                          epochs = epochs,
                                          batch_size = self.n_batch,
                                          validation_data = (self.X_val, self.y_val),
                                          # class_weight = class_weights_dict,
                                          shuffle = True)
                                        
        bce_history.append(self.training_history.history)

        # print(training_history.history.keys())

        if not os.path.exists("saved_model"):
            os.mkdir("saved_model")

            self.model.save("saved_model/rnn_model.keras")

        return bce_history, self.model, self.training_history
    
    def visualize_results(self):
        
        # Plot history for accuracies
        plt.plot(self.training_history.history["accuracy"], color = "purple")
        plt.plot(self.training_history.history["val_accuracy"], color = "green")
        plt.title("Model Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Test"], loc = "lower right")
        plt.show()

        # Plot history for losses
        plt.plot(self.training_history.history["loss"], color = "purple")
        plt.plot(training_history.history["val_loss"], color = "green")
        plt.title("Model Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Test"], loc = "upper right")
        plt.show()
    
    def eval_model_without_lures(self):

        eval_results = self.model.evaluate(self.X_test, self.y_test, batch_size = 128)
        print(f"Overall Test Loss, Test Accuracy: {eval_results}")

        predictions = self.model.predict(self.X_test)

        predicted_responses = (predictions >= 0.5).astype(int).flatten()

        for i in range(10):
            letters_sequence = X_test[i]

            true_response = y_test[i]

            predicted_response = predicted_responses[i]

            print(f"Trial {i + 1}:")
            print(f"Letters : {letters_sequence}")
            print(f"True Response: {'target' if true_response == 1 else 'nontarget'}")
            print(f"Predicted Response: {'target' if predicted_response == 1 else 'nontarget'}")
            print("-" * 50)
        
        return eval_results

    def eval_model_with_lures(self):

        predictions_with_lures = self.model.predict(self.X_test_with_lures)
        predicted_labels = (predictions_with_lures >= 0.5).astype(int).flatten()

        # print("All true labels in y_test_with_lures:", self.y_test_with_lures)

        lure_count = 0 
        for i, (pred, true_label) in enumerate(zip(predicted_labels, self.y_test_with_lures)):
            if true_label == 2:  
                lure_count += 1
                letters_sequence = self.X_test_with_lures[i]
                predicted_response = 'target' if pred == 1 else 'nontarget'

                print(f"Lure Trial {i + 1}:")
                print(f"Letters (sequence): {letters_sequence}")
                print(f"True Label: lure")
                print(f"Predicted Response: {predicted_response}")
                print("-" * 50)
        
        print(f"Total lure trials found: {lure_count}")

#%%

if __name__ == "__main__":

    data_processor = DataPreprocessor(data_path_bin_cl = "3-back task/raw_data_with_lure.csv", data_path_multiclass_pred = "3-back task/raw_data_with_lure_test.csv")  
    preprocessed_data_bin_cl = data_processor.preprocess_data_for_bin_cl()
    preprocessed_data_multic_pred = data_processor.preprocess_data_for_multic_pred()

    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.split_data_for_bin_pred(preprocessed_data_bin_cl)
    X_test_with_lures, y_test_with_lures = data_processor.split_data_for_multic_pred(preprocessed_data_multic_pred)

    print("Training dataset shape:", X_train.shape, y_train.shape)
    print("Validation dataset shape:", X_val.shape, y_val.shape)
    print("Test dataset shape:", X_test.shape, y_test.shape)
    print("X test with lures shape", X_test_with_lures.shape)
    print("y test with lures shape", y_test_with_lures.shape)

    lstm_trainer = LSTMTrainer(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, X_test_with_lures = X_test_with_lures, y_test_with_lures = y_test_with_lures, n_batch = 64, learning_rate = 0.01)
    lstm_trainer.initialize_model()
    bce_history, model, training_history = lstm_trainer.train_model(epochs = 100)
    display_acc_loss = lstm_trainer.visualize_results()
    eval_results_wo_lures = lstm_trainer.eval_model_without_lures()
    eval_results_with_lures = lstm_trainer.eval_model_with_lures()

# %%
