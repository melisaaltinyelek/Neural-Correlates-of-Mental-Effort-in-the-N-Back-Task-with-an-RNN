#%%

from RNN_model import DataPreprocessor
import tensorflow as tf
from tensorflow import keras
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

class PreprocessBehavioralData():
    def __init__(self, loaded_model, behavioral_3back_data):
        self.loaded_model = loaded_model
        self.behavioral_3back_data = behavioral_3back_data
        self.df_3back = None
    
    def preprocess_beh_data(self):
        
        df_3back = pd.read_csv(self.behavioral_3back_data, sep = ";")

        letter_encoder = OneHotEncoder(sparse_output = False)
        encoded_letters = letter_encoder.fit_transform(df_3back[["letter"]])
        df_3back["letter"] = encoded_letters.tolist()

        self.df_3back = df_3back

        # df_3back.to_csv("preprocessed_3back_data.csv", index = False, sep = ",")

        return self.df_3back
    
    def create_seq_for_3back(self, df_3back, n_steps = 4):

         X_test_beh, y_test_beh = DataPreprocessor.create_sequences(None, df_3back, n_steps = n_steps)

         print(f"This is X_test_beh: {X_test_beh}")
         print(f"This is y_test_beh: {y_test_beh}")

         return X_test_beh, y_test_beh

#%% 

if __name__ == "__main__":

    loaded_model = tf.keras.models.load_model("saved_model/rnn_model.keras")
    # print(loaded_model)

    data_preprocessor = PreprocessBehavioralData(
        loaded_model = loaded_model,
        behavioral_3back_data = "3-back data/3back_beh_data.csv")
    
    preprocessed_data = data_preprocessor.preprocess_beh_data()
    X_test_beh, y_test_beh = data_preprocessor.create_seq_for_3back(preprocessed_data, n_steps = 4)
    
#%%
