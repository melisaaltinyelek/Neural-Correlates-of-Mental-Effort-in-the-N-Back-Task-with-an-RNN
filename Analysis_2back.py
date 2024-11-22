#%%

from RNN_model import DataPreprocessor, LSTMTrainer
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
from ast import literal_eval
import pandas as pd
import numpy as np
import os
import glob

#%%

class DataPreprocessor_2back():
    def __init__(self, data_path_2back_bin, data_path_2back_mc):

        self.data_path_2back_bin = data_path_2back_bin
        self.data_path_2back_mc = data_path_2back_mc
        self.df = None

    def prep_2back_data_wo_lures(self):

        self.df = DataPreprocessor.preprocess_data(self, df = self.data_path_2back_bin,
                                                            mode = "binary",
                                                            lure_replacement = "nontarget",
                                                            output_path = "2-back data/test_nback_data_without_lure.csv")

        return self.df
    
    def prep_2back_data_w_lures(self):
        pass

    def create_sequence_and_split_data(self):

        self.X_test_2back_wo_lures, self.y_test_2back_wo_lures = DataPreprocessor.create_sequences(self, df = self.df, n_steps = 3)

        # print(f"This is the X (letters): {X_test_2back_wo_lures}")
        # print(f"This is the y (responses): {y_test_2back_wo_lures}")

        return self.X_test_2back_wo_lures, self.y_test_2back_wo_lures
    
class AnalyzeRNNon2back():
    def __init__(self):
        self.saved_model = tf.keras.models.load_model("saved_model/rnn_model.keras")

    def eval_model(self, X_test, y_test):

        eval_results = self.saved_model.evaluate(X_test, y_test, batch_size=128)
        print(f"Overall Test Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}")

        predictions = self.saved_model.predict(X_test)
        pred_resp = (predictions >= 0.5).astype(int).flatten()

        for i in range(10):
            letters_sequence = X_test[i]

            true_response = y_test[i]
            predicted_response = pred_resp[i]

            print(f"Trial {i + 1}:")
            print(f"Letters : {letters_sequence}")
            print(f"True Response: {'target' if true_response == 1 else 'nontarget'}")
            print(f"Predicted Response: {'target' if predicted_response == 1 else 'nontarget'}")
            print("-" * 50)

        return pred_resp

#%%

if __name__ == "__main__":

    data_preprocessor = DataPreprocessor_2back(data_path_2back_bin = "2-back data/raw_data_with_lure.csv", data_path_2back_mc = None)
    data_preprocessor.prep_2back_data_wo_lures()
    X_test_2_back_wo_lures, y_test_2back_wo_lures = data_preprocessor.create_sequence_and_split_data()

    rnn_model = AnalyzeRNNon2back()
    rnn_model.eval_model(X_test_2_back_wo_lures, y_test_2back_wo_lures)
#%%