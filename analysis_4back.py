#%%

from RNN_model import DataPreprocessor, RNNTrainer
from save_and_plot_accuracies import save_acc_to_json
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%

class DataPreprocessor_4back():
    def __init__(self):

        self.df_wo_lures = None
        self.df_w_lures = None

    def prep_4back_data_wo_lures(self, data_path_4back_bin):

        self.df_wo_lures = DataPreprocessor.preprocess_data(self, df = data_path_4back_bin,
                                                            mode = "binary",
                                                            output_path = "4-back data/test_4back_data_wo_lures.csv")

        return self.df_wo_lures
    
    def prep_4back_data_w_lures(self, data_path_4back_mc):
        
        self.df_w_lures = DataPreprocessor.preprocess_data(self, df = data_path_4back_mc,
                                                           mode = "multiclass",
                                                           output_path = "4-back data/test_4back_data2_w_lures.csv")
        
        return self.df_w_lures

    def create_seq_and_split_data(self):

        self.X_test_4back_wo_lures, self.y_test_4back_wo_lures = DataPreprocessor.create_sequences(self, df = self.df_wo_lures, n_steps = 5)
       
        return self.X_test_4back_wo_lures, self.y_test_4back_wo_lures
    
    def create_seq_and_split_lure_data(self):

        self.X_test_4back_w_lures, self.y_test_4back_w_lures = DataPreprocessor.create_sequences(self, df = self.df_w_lures, n_steps = 5)
        
        return self.X_test_4back_w_lures, self.y_test_4back_w_lures
    
class AnalyzeRNNon4backData():
    def __init__(self, rnn_trainer):

        self.rnn_trainer = rnn_trainer
        self.saved_model = tf.keras.models.load_model("saved_model/rnn_model.keras")
        self.saved_model.summary()
    
    def eval_model_without_lures(self, X_test, y_test, n_back):

        test_acc_wo_lures, predicted_responses = self.rnn_trainer.eval_model_wo_lures(X_test, y_test, n_back, self.saved_model)
        
        self.pred_resp = predicted_responses
        self.test_acc = test_acc_wo_lures
        
        return test_acc_wo_lures, predicted_responses
    
    def visualize_preds_without_lures(self, y_test, predicted_responses, n_back):
        
        self.rnn_trainer.visualize_preds_wo_lures(y_test, predicted_responses, n_back)
    
    def eval_model_with_lures(self, X_test_w_lures, y_test_w_lures, n_back):

        test_acc_w_lures, pred_resp_w_lures = self.rnn_trainer.eval_model_w_lures(X_test_w_lures, y_test_w_lures, n_back, self.saved_model)

        self.pred_resp_w_lures = pred_resp_w_lures
        self.test_acc_w_lures = test_acc_w_lures

        return test_acc_w_lures, pred_resp_w_lures
    
    def visualize_preds_with_lures(self, y_test_w_lures, pred_responses_w_lures, n_back):

        self.rnn_trainer.visualize_preds_w_lures(y_test_w_lures, pred_responses_w_lures, n_back)

    def visualize_embeddings(self, X_test_w_lures, y_test_w_lures, n_back):

        self.rnn_trainer.create_submodel(X_test_w_lures, y_test_w_lures, n_back)

#%%

if __name__ == "__main__":

    data_preprocessor = DataPreprocessor_4back()
    data_preprocessor.prep_4back_data_wo_lures(data_path_4back_bin = "4-back data/training_4back_data_wo_lures.csv")
    data_preprocessor.prep_4back_data_w_lures(data_path_4back_mc = "4-back data/training_4back_data_w_lures.csv")

    X_test_4back_wo_lures, y_test_4back_wo_lures = data_preprocessor.create_seq_and_split_data() 
    X_test_4back_w_lures, y_test_4back_w_lures = data_preprocessor.create_seq_and_split_lure_data()

    rnn_model = AnalyzeRNNon4backData(
        rnn_trainer = RNNTrainer(
            X_train = None, y_train = None, X_val = None, y_val = None, 
            X_test = None, y_test = None, X_test_w_lures = None, y_test_w_lures = None,
            n_batch = None, learning_rate = None
    ))

    test_acc_wo_lures, pred_responses = rnn_model.eval_model_without_lures(X_test_4back_wo_lures, y_test_4back_wo_lures, 4)
    rnn_model.visualize_preds_without_lures(y_test_4back_wo_lures, pred_responses, 4)
    
    test_acc_w_lures, pred_responses_w_lures = rnn_model.eval_model_with_lures(X_test_4back_w_lures, y_test_4back_w_lures, 4)
    rnn_model.visualize_preds_with_lures(y_test_4back_w_lures, pred_responses_w_lures, 4)

    save_acc_to_json("4-back", test_acc_wo_lures, test_acc_w_lures)

    rnn_model.visualize_embeddings(X_test_4back_w_lures, y_test_4back_w_lures, 4)
    #%%