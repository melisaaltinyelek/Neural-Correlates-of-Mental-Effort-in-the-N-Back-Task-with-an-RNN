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

        self.df = DataPreprocessor.preprocess_data_for_bin_cl(self, df = self.data_path_2back_bin,
                                                                lure_replacement = "nontarget",
                                                                output_path = "2-back data/nback_data_without_lure.csv")

        return self.df
    
    def prep_2back_data_w_lures(self):
        pass

#%%

if __name__ == "__main__":

    # loaded_model = tf.keras.models.load_model("saved_model/rnn_model.keras")
    # print(loaded_model)

    data_preprocessor = DataPreprocessor_2back(data_path_2back_bin = "2-back data/raw_data_with_lure.csv", data_path_2back_mc = None)
    data_preprocessor.prep_2back_data_wo_lures()
#%%
