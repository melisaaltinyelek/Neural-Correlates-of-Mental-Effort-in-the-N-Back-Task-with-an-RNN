#%%

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

class DataPreprocessor:
    def __init__(self):

        self.df = None
    
    def preprocess_data(self, df, mode = "binary", lure_replacement = "nontarget", output_path = None):

        df = pd.read_csv(df)

        if mode == "binary":
            if lure_replacement:
                df.replace(to_replace = "lure", value = lure_replacement, inplace = True)

            response_mapping = {
                "nontarget": 0,
                "target": 1
            }

        elif mode == "multiclass":
            response_mapping = {
            "nontarget": 0,
            "target": 1,
            "lure": 2
        }

        df["response"] = df["response"].map(response_mapping)

        letter_encoder = OneHotEncoder(sparse_output = False) # sparse_output = False returns a NumPy array
        encoded_letters = letter_encoder.fit_transform(df[["letter"]])

        # Convert the encoded columns to lists to store them in pandas DataFrame
        df["letter"] = encoded_letters.tolist()
        
        self.df = df

        # if output_path:
        #     df.to_csv(output_path, index = False)

        print(self.df)

        return self.df
    
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

        self.training_history = self.model.fit(self.X_train, self.y_train,
                                          epochs = epochs,
                                          batch_size = self.n_batch,
                                          validation_data = (self.X_val, self.y_val),
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
        
    def eval_model_without_lures(self, X_test, y_test):

    
        eval_results = self.model.evaluate(X_test, y_test, batch_size = 128)
        print(f"Overall Test Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}")

        predictions = self.model.predict(X_test)
        predicted_responses = (predictions >= 0.5).astype(int).flatten()

        print(f"The predicted responses are: {predicted_responses}")

        for i in range(10):
            letters_sequence = X_test[i]

            true_response = y_test[i]

            predicted_response = predicted_responses[i]

            print(f"Trial {i + 1}:")
            print(f"Letters : {letters_sequence}")
            print(f"True Response: {'target' if true_response == 1 else 'nontarget'}")
            print(f"Predicted Response: {'target' if predicted_response == 1 else 'nontarget'}")
            print("-" * 50)

        self.predicted_responses = predicted_responses
        
        return self.predicted_responses 
    
    def visualize_wo_lures(self):
        
        # Calculate the total number of target trials and correctly predicted target trials
        self.num_target_wo_lures = 0
        self.num_corr_pred_target_wo_lures = 0

        for i in range(len(self.y_test)):
            target_response = self.y_test[i]
            pred_target_resp = self.predicted_responses[i]

            if target_response == 1:
                self.num_target_wo_lures += 1

            if target_response == 1: 
                if pred_target_resp == target_response:
                    self.num_corr_pred_target_wo_lures += 1

        # print(f"The number target trials: {self.num_target_trials}")
        # print(f"The number of correctly predicted target trials: {self.num_corr_pred_target_trials}")

        # Calculate the total number of nontarget trials and correctly predicted nontarget trials
        self.num_nontarget_wo_lures = 0
        self.num_corr_pred_nontarget_wo_lures = 0

        for i in range(len(self.y_test)):
            nontarget_response = self.y_test[i]
            pred_nontarget_resp = self.predicted_responses[i]

            if nontarget_response == 0:
                self.num_nontarget_wo_lures += 1

            if nontarget_response == 0:
                if pred_nontarget_resp == nontarget_response:
                    self.num_corr_pred_nontarget_wo_lures += 1

        # print(f"The number of nontarget trials: {self.num_nontarget_trials}")
        # print(f"The number of correctly predicted nontarget trials: {self.num_corr_pred_nontarget_trials}")

        # Calculate the accuracies for both nontarget and target trials for visualization
        acc_target = self.num_corr_pred_target_wo_lures / self.num_target_wo_lures
        acc_nontarget = self.num_corr_pred_nontarget_wo_lures / self.num_nontarget_wo_lures

        print(f"Accuracy score for the target trials: {acc_target}")
        print(f"Accuracy score for the nontarget trials: {acc_nontarget}")
        
        labels = ["nontarget", "target"]
        acc_list = [acc_nontarget, acc_target]

        plt.figure(figsize = (6, 7))
        plt.bar(labels, acc_list, color = ["#C3B1E1", "#77DD77"])
        plt.xlabel("Response Categories")
        plt.ylabel("Accuracy")
        plt.title("Model Prediction Across Trial Types")
        plt.show()

        confusion_matrix_wo_lures = metrics.confusion_matrix(self.y_test, self.predicted_responses)
        cm_display_wo_lures = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_wo_lures, display_labels = [0, 1])
        cm_display_wo_lures.plot()
        plt.show()

    def eval_model_with_lures(self, X_test_with_lures, y_test_with_lures):

        self.predictions_w_lures = self.model.predict(X_test_with_lures)
        self.pred_resp_w_lures = (self.predictions_w_lures >= 0.5).astype(int).flatten()

        print("All true labels in y_test_with_lures:", y_test_with_lures)

        lure_count = 0 
        for i, (pred, true_label) in enumerate(zip(self.pred_resp_w_lures, y_test_with_lures)):
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

        return self.pred_resp_w_lures

    def visualize_w_lures(self):

        self.num_corr_pred_nontarget_w_lures = 0
        self.num_corr_pred_target_w_lures = 0
        self.num_corr_pred_lure = 0

        self.num_incorr_pred_nontarget_as_target = 0
        self.num_incorr_pred_nontarget_as_lure = 0

        self.num_incorr_pred_target_as_nontarget = 0
        self.num_incorr_pred_target_as_lure = 0

        self.num_incorr_pred_lure_as_nontarget = 0
        self.num_incorr_pred_lure_as_target = 0

        for i in range(len(self.y_test_with_lures)):
            true_response = self.y_test_with_lures[i]
            pred_target_resp = self.pred_resp_w_lures[i]

            if true_response == 0 and true_response == pred_target_resp:
                self.num_corr_pred_nontarget_w_lures += 1
            elif true_response == 1 and true_response == pred_target_resp:
                self.num_corr_pred_target_w_lures += 1
            elif true_response == 2 and true_response == pred_target_resp:
                self.num_corr_pred_lure += 1

            if true_response == 0 and pred_target_resp == 1:
                self.num_incorr_pred_nontarget_as_target += 1
            elif true_response == 0 and pred_target_resp == 2:
                self.num_incorr_pred_nontarget_as_lure += 1
            elif true_response == 1 and pred_target_resp == 0:
                self.num_incorr_pred_target_as_nontarget += 1
            elif true_response == 1 and pred_target_resp == 2:
                self.num_incorr_pred_target_as_lure += 1
            elif true_response == 2 and pred_target_resp == 0:
                self.num_incorr_pred_lure_as_nontarget += 1
            elif true_response == 2 and pred_target_resp == 1:
                self.num_incorr_pred_lure_as_target += 1

        all_labels = ["nontarget", "target", "lure"]

        corr_classifications = [
            self.num_corr_pred_nontarget_w_lures,
            self.num_corr_pred_target_w_lures,
            self.num_corr_pred_lure
            ]
        
        misclassifications = [
            self.num_incorr_pred_target_as_nontarget,
            self.num_incorr_pred_lure_as_nontarget,
            self.num_incorr_pred_nontarget_as_target,
            self.num_incorr_pred_lure_as_target,
            self.num_incorr_pred_nontarget_as_lure,
            self.num_incorr_pred_target_as_lure
             ]

        misclassified_as_target_for_nontarget = misclassifications[2]  # Misclassified as target when true label is nontarget
        misclassified_as_lure_for_nontarget = misclassifications[1]    # Misclassified as lure when true label is nontarget

        misclassified_as_nontarget_for_target = misclassifications[0]  # Misclassified as nontarget when true label is target
        misclassified_as_lure_for_target = misclassifications[3]       # Misclassified as lure when true label is target

        misclassified_as_nontarget_for_lure = misclassifications[4]    # Misclassified as nontarget when true label is lure
        misclassified_as_target_for_lure = misclassifications[5]       # Misclassified as target when true label is lure

        x = np.arange(len(all_labels))
        bar_width = 0.50

        plt.figure(figsize = (10, 6))
     
        plt.bar(x[0], misclassified_as_target_for_nontarget, width = bar_width, color = "#22CE83", label = "Misclassified as target")
        plt.bar(x[0], misclassified_as_lure_for_nontarget, width = bar_width, bottom = misclassified_as_target_for_nontarget, color = "#C3B1E1", label = "Misclassified as lure")

        plt.bar(x[1], misclassified_as_nontarget_for_target, width = bar_width, color = "#FFB7CE", label = "Misclassified as nontarget")
        plt.bar(x[1], misclassified_as_lure_for_target, width = bar_width, bottom = misclassified_as_nontarget_for_target, color = "#38ACEC", label = "Misclassified as lure")

        plt.bar(x[2], misclassified_as_nontarget_for_lure, width = bar_width, color = "#FFEF00", label = "Misclassified as nontarget")
        plt.bar(x[2], misclassified_as_target_for_lure, width = bar_width, bottom = misclassified_as_nontarget_for_lure, color = "#FF7F50", label = "Misclassified as target")

        plt.xticks(x, all_labels)
        plt.xlabel("Trial Type")
        plt.ylabel("Number of Trials")
        plt.title("Misclassifications Across Trial Types")
        plt.legend(loc = "upper right")
        plt.tight_layout()
        plt.show()

        confusion_matrix_w_lures = metrics.confusion_matrix(self.y_test_with_lures, self.pred_resp_w_lures)
        cm_display_w_lures = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_w_lures, display_labels = [0, 1, 2])
        cm_display_w_lures.plot()
        plt.show()

#%%

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()

    processed_binary_df = data_preprocessor.preprocess_data(
        df = "3-back data/raw_data_with_lure.csv",
        mode = "binary",
        lure_replacement = "nontarget",
        output_path = "3-back data/test_nback_data_without_lure.csv"
    )

    processed_multiclass_df = data_preprocessor.preprocess_data(
        df = "3-back data/raw_data_with_lure_test.csv",
        mode = "multiclass",
        output_path = "3-back data/test_nback_data_with_lure.csv"
    )
     
    X_train, y_train, X_val, y_val, X_test, y_test = data_preprocessor.split_data_for_bin_pred(processed_binary_df)
    X_test_with_lures, y_test_with_lures = data_preprocessor.split_data_for_multic_pred(processed_multiclass_df)

    print("Training dataset shape:", X_train.shape, y_train.shape)
    print("Validation dataset shape:", X_val.shape, y_val.shape)
    print("Test dataset shape:", X_test.shape, y_test.shape)
    print("X test with lures shape", X_test_with_lures.shape)
    print("y test with lures shape", y_test_with_lures.shape)

    lstm_trainer = LSTMTrainer(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, X_test_with_lures = X_test_with_lures, y_test_with_lures = y_test_with_lures, n_batch = 64, learning_rate = 0.01)
    lstm_trainer.initialize_model()
    bce_history, model, training_history = lstm_trainer.train_model(epochs = 100)
    lstm_trainer.visualize_results()
    lstm_trainer.eval_model_without_lures(X_test, y_test)
    lstm_trainer.visualize_wo_lures()
    lstm_trainer.eval_model_with_lures(X_test_with_lures, y_test_with_lures)
    lstm_trainer.visualize_w_lures()

#%%