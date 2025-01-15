#%%

import tensorflow as tf
from tensorflow.keras.models import Model
import keras
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#%%

class DataPreprocessor:
    def __init__(self):

        self.df = None
    
    def preprocess_data(self, df, mode = "binary", output_path = None):

        df = pd.read_csv(df)

        if mode == "binary":
     
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
    
    def split_data_for_mc_pred(self, data):

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

class RNNTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, X_test_w_lures, y_test_w_lures, n_batch, learning_rate):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_w_lures = X_test_w_lures
        self.y_test_w_lures = y_test_w_lures
        self.n_batch = n_batch
        self.learning_rate = learning_rate
        self.predicted_responses = None
        self.model = None

    def initialize_model(self):

        self.model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(1, input_shape = (1, self.X_train.shape[2])),
            tf.keras.layers.Dense(2, activation = "tanh"),
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
    
    def visualize_training_results(self):
        
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
        
    def eval_model_wo_lures(self, X_test, y_test, n_back, model = None):

        model_to_use = model if model else self.model

        eval_results = model_to_use.evaluate(X_test, y_test)
        test_acc = f"{eval_results[1]:.2f}"
        # print(test_acc)
        print(f"Overall Test Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}")

        predictions = model_to_use.predict(X_test)
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

        print("Classification Report:")
        print(classification_report(y_test, predicted_responses, target_names = ["nontarget", "target"]))
        
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize = (10, 6))
        plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw = 2, linestyle = '--', label = "Chance Level")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        title = f"Binary Classification ROC: {n_back}-Back Task"
        plt.title(title)
        plt.legend(loc = "lower right")
        plt.show()

        self.predicted_responses = predicted_responses

        return test_acc, predicted_responses
    
    def visualize_preds_wo_lures(self, y_test, predicted_responses, n_back):
           
        # Calculate the total number of target trials and correctly predicted target trials
        num_targets_wo_lures = 0
        num_corr_pred_targets_wo_lures = 0

        for i in range(len(y_test)):
            target_response = y_test[i]
            pred_target_resp = predicted_responses[i]

            if target_response == 1:
                num_targets_wo_lures += 1

            if target_response == 1: 
                if pred_target_resp == target_response:
                    num_corr_pred_targets_wo_lures += 1

        # Calculate the total number of nontarget trials and correctly predicted nontarget trials
        num_nontargets_wo_lures = 0
        num_corr_pred_nontargets_wo_lures = 0

        for i in range(len(y_test)):
            nontarget_response = y_test[i]
            pred_nontarget_resp = predicted_responses[i]

            if nontarget_response == 0:
                num_nontargets_wo_lures += 1

            if nontarget_response == 0:
                if pred_nontarget_resp == nontarget_response:
                    num_corr_pred_nontargets_wo_lures += 1

        # Calculate the accuracies for both nontarget and target trials for visualization
        target_acc = num_corr_pred_targets_wo_lures / num_targets_wo_lures
        nontarget_acc = num_corr_pred_nontargets_wo_lures / num_nontargets_wo_lures

        labels = ["nontarget", "target"]
        acc_list = [nontarget_acc, target_acc]

        plt.figure(figsize = (6, 7))
        plt.bar(labels, acc_list, color = ["#C3B1E1", "#77DD77"])
        plt.xlabel("Response Categories")
        plt.ylabel("Accuracy")
        title = f"Model Predictions Over Trials: {n_back}-Back Binary Classification"
        plt.title(title)
        plt.show()

        confusion_matrix_wo_lures = metrics.confusion_matrix(y_test, predicted_responses)
        cm_display_wo_lures = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_wo_lures, display_labels = [0, 1])
        cm_title = f"Confusion Matrix for {n_back}-Back Task (Without Lures)"
        ax = cm_display_wo_lures.plot().ax_
        ax.set_title(cm_title)
        plt.show()

        print(f"The number of nontarget trials: {num_nontargets_wo_lures}")
        print(f"The number of correctly predicted nontarget trials: {num_corr_pred_nontargets_wo_lures}")

        print(f"The number target trials: {num_targets_wo_lures}")
        print(f"The number of correctly predicted target trials: {num_corr_pred_targets_wo_lures}")

        print(f"Accuracy score for the target trials: {target_acc}")
        print(f"Accuracy score for the nontarget trials: {nontarget_acc}")

    def eval_model_w_lures(self, X_test_w_lures, y_test_w_lures, n_back, model = None):

        model_to_use = model if model else self.model

        eval_results = model_to_use.evaluate(X_test_w_lures, y_test_w_lures)
        test_acc_w_lures = f"{eval_results[1]:.2f}"
        # print(test_acc)
        print(f"Overall Test Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}")

        predictions_w_lures = model_to_use.predict(X_test_w_lures)
        pred_resp_w_lures = (predictions_w_lures >= 0.5).astype(int).flatten()

        print("All true labels in y_test_with_lures:", y_test_w_lures)

        lure_count = 0 
        for i, (pred, true_label) in enumerate(zip(pred_resp_w_lures, y_test_w_lures)):
            if true_label == 2:  
                lure_count += 1
                letters_sequence = X_test_w_lures[i]
                predicted_response = "target" if pred == 1 else "nontarget"

                print(f"Lure Trial {i + 1}:")
                print(f"Letters (sequence): {letters_sequence}")
                print(f"True Label: lure")
                print(f"Predicted Response: {predicted_response}")
                print("-" * 50)
        
        print(f"Total lure trials found: {lure_count}")

        print("Classification Report (with lures):")
        print(classification_report(y_test_w_lures, pred_resp_w_lures, target_names = ["nontarget", "target", "lure"]))

        classes = [0, 1, 2]
        y_test_binarized = label_binarize(y_test_w_lures, classes = [0, 1, 2])
        predictions_w_lures_expanded = np.zeros((len(predictions_w_lures), 3)) 

        predictions_w_lures_expanded[:, 1] = predictions_w_lures.flatten()  

        predictions_w_lures_expanded[:, 0] = 1 - predictions_w_lures.flatten() 

        predictions_w_lures_expanded[:, 2] = 0 

        plt.figure(figsize = (10, 6))

        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], predictions_w_lures_expanded[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw = 2, label = f"Class {class_label} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--", label = "Chance Level")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        title = f"Multiclass Classification ROC: {n_back}-Back Task"
        plt.title(title)
        plt.legend(loc = "lower right")
        plt.show()

        # print("Shape of predictions_w_lures:", predictions_w_lures.shape)
        # print("Shape of y_test_binarized:", y_test_binarized.shape)
        
        self.pred_resp_w_lures = pred_resp_w_lures

        return test_acc_w_lures, pred_resp_w_lures

    def visualize_preds_w_lures(self, y_test_w_lures, pred_resp_w_lures, n_back):

        num_nontargets_w_lures = 0
        num_targets_w_lures = 0
        num_lures = 0

        num_corr_pred_nontargets_w_lures = 0
        num_corr_pred_targets_w_lures = 0
        num_corr_pred_lures = 0

        num_incorr_pred_nontargets_as_target = 0
        num_incorr_pred_nontargets_as_lure = 0

        num_incorr_pred_targets_as_nontarget = 0
        num_incorr_pred_targets_as_lure = 0

        num_incorr_pred_lures_as_nontarget = 0
        num_incorr_pred_lures_as_target = 0

        for i in range(len(y_test_w_lures)):

            nontarget_resp_w_lures = y_test_w_lures[i]

            if nontarget_resp_w_lures == 0:
                num_nontargets_w_lures += 1

            if nontarget_resp_w_lures == 1:
                num_targets_w_lures += 1

            if nontarget_resp_w_lures == 2:
                num_lures += 1

        for i in range(len(y_test_w_lures)):
            
            true_response = y_test_w_lures[i]
            pred_target_resp = pred_resp_w_lures[i]

            if true_response == 0 and true_response == pred_target_resp:
                num_corr_pred_nontargets_w_lures += 1
            elif true_response == 1 and true_response == pred_target_resp:
                num_corr_pred_targets_w_lures += 1
            elif true_response == 2 and true_response == pred_target_resp:
                num_corr_pred_lures += 1

            if true_response == 0 and pred_target_resp == 1:
                num_incorr_pred_nontargets_as_target += 1
            elif true_response == 0 and pred_target_resp == 2:
                num_incorr_pred_nontargets_as_lure += 1
            elif true_response == 1 and pred_target_resp == 0:
                num_incorr_pred_targets_as_nontarget += 1
            elif true_response == 1 and pred_target_resp == 2:
                num_incorr_pred_targets_as_lure += 1
            elif true_response == 2 and pred_target_resp == 0:
                num_incorr_pred_lures_as_nontarget += 1
            elif true_response == 2 and pred_target_resp == 1:
                num_incorr_pred_lures_as_target += 1

        # Calculate the accuracies for nontarget, target and lure 
        target_acc_w_lures = num_corr_pred_targets_w_lures / num_targets_w_lures
        nontarget_acc_w_lures = num_corr_pred_nontargets_w_lures / num_nontargets_w_lures
        lure_acc = num_corr_pred_lures / num_lures

        all_labels = ["nontarget", "target", "lure"]

        corr_classifications = [
            num_corr_pred_nontargets_w_lures,
            num_corr_pred_targets_w_lures,
            num_corr_pred_lures
            ]
        
        misclassifications = [
            num_incorr_pred_targets_as_nontarget,
            num_incorr_pred_lures_as_nontarget,
            num_incorr_pred_nontargets_as_target,
            num_incorr_pred_lures_as_target,
            num_incorr_pred_nontargets_as_lure,
            num_incorr_pred_targets_as_lure
             ]

        misclassified_as_target_for_nontarget = misclassifications[2]  # Misclassified as target when true label is nontarget
        misclassified_as_lure_for_nontarget = misclassifications[1]    # Misclassified as lure when true label is nontarget

        misclassified_as_nontarget_for_target = misclassifications[0]  # Misclassified as nontarget when true label is target
        misclassified_as_lure_for_target = misclassifications[3]       # Misclassified as lure when true label is target

        misclassified_as_nontarget_for_lure = misclassifications[4]    # Misclassified as nontarget when true label is lure
        misclassified_as_target_for_lure = misclassifications[5]       # Misclassified as target when true label is lure


        label_colors = {
            "Misclassified as nontarget": "#FFB7CE",
            "Misclassified as target": "#22CE83",  
            "Misclassified as lure": "#9172EC",  
        }

        x = np.arange(len(all_labels))
        bar_width = 0.50

        plt.figure(figsize = (10, 6))
     
        plt.bar(x[0], misclassified_as_target_for_nontarget, width = bar_width, color = label_colors["Misclassified as target"], label = "Misclassified as target")
        plt.bar(x[0], misclassified_as_lure_for_nontarget, width = bar_width, bottom = misclassified_as_target_for_nontarget, color = label_colors["Misclassified as lure"], label = "Misclassified as lure")

        plt.bar(x[1], misclassified_as_nontarget_for_target, width = bar_width, color = label_colors["Misclassified as nontarget"], label = "Misclassified as nontarget")
        plt.bar(x[1], misclassified_as_lure_for_target, width = bar_width, bottom = misclassified_as_nontarget_for_target, color = label_colors["Misclassified as lure"], label = "Misclassified as lure")

        plt.bar(x[2], misclassified_as_nontarget_for_lure, width = bar_width, color = label_colors["Misclassified as nontarget"], label = "Misclassified as nontarget")
        plt.bar(x[2], misclassified_as_target_for_lure, width = bar_width, bottom = misclassified_as_nontarget_for_lure, color = label_colors["Misclassified as target"], label = "Misclassified as target")

        plt.xticks(x, all_labels)
        plt.xlabel("Correct Labels")
        plt.ylabel("Number of Trials")
        title = f"Model Predictions Over Trials: {n_back}-Back Multiclass Classification"
        plt.title(title)

        custom_legend = [
            plt.Line2D([0], [0], color = label_colors["Misclassified as nontarget"], lw = 10, label = "nontarget"),
            plt.Line2D([0], [0], color = label_colors["Misclassified as target"], lw = 10, label = "target"),
            plt.Line2D([0], [0], color = label_colors["Misclassified as lure"], lw = 10, label = "lure")
        ]

        plt.legend(handles = custom_legend, loc = "upper right", title = "Trial Types", frameon = False)
        plt.tight_layout()
        plt.show()

        confusion_matrix_w_lures = metrics.confusion_matrix(y_test_w_lures, pred_resp_w_lures)
        cm_display_w_lures = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_w_lures, display_labels = [0, 1, 2])
        cm_title = f"Confusion Matrix for {n_back}-Back Task (With Lures)"
        ax = cm_display_w_lures.plot().ax_
        ax.set_title(cm_title)
        plt.show()

        print(f"The number of nontarget trials: {num_nontargets_w_lures}")
        print(f"The number of correctly predicted nontarget trials: {num_corr_pred_nontargets_w_lures}")

        print(f"The number of target trials: {num_targets_w_lures}")
        print(f"The number of correctly predicted target trials: {num_corr_pred_targets_w_lures}")

        print(f"The number of lure trials: {num_lures}")
        print(f"The number of correctly predicted lure trials: {num_corr_pred_lures}")

        print(f"Accuracy score for the nontarget trials: {nontarget_acc_w_lures}")
        print(f"Accuracy score for the target trials: {target_acc_w_lures}")
        print(f"Accuracy score for the lure trials: {lure_acc}")
    
    def create_submodel(self, X_test_w_lures, y_test_w_lures, n_back):

        trained_model = tf.keras.models.load_model("saved_model/rnn_model.keras")

        for i, layer in enumerate(trained_model.layers):
            print(f"Layer {i}: {layer.name}, Type: {type(layer)}")

        input_shape = tf.keras.layers.Input(shape = (None, trained_model.input_shape[2]))
        print(f"The input shape: {input_shape}")

        rnn_output = trained_model.layers[0](input_shape)
        print(f"The RNN layer output shape (embedding): {rnn_output}")

        embedding_model = Model(
            inputs = input_shape,
            outputs = rnn_output
            )

        embeddings = embedding_model.predict(X_test_w_lures)

        print("Embeddings shape:", embeddings.shape)
        print(f"Embeddings length: {len(embeddings)}")
        print("Embeddings data type:", embeddings.dtype)
        print("Sample embeddings:", embeddings[:5])  

        target_embeddings = []
        lure_embeddings = []

        for embedding, true_label in zip(embeddings, y_test_w_lures):
            if true_label == 1:
                target_embeddings.append(embedding)
            elif true_label == 2:
                lure_embeddings.append(embedding)

        print(f"Target embeddings: {target_embeddings}")
        print(f"Lure embeddings: {lure_embeddings}")
    
        target_embeddings = np.array(target_embeddings)
        lure_embeddings = np.array(lure_embeddings)

        plt.hist(target_embeddings, bins = 50, alpha = 0.6, label = "Targets", color = "#14A3C7")
        plt.hist(lure_embeddings, bins = 50, alpha = 0.4, label = "Lures", color = "#7B68EE")

        title = f"Embedding Distributions for {n_back}-Back Task"
        plt.title(title)
        plt.xlabel("Embedding values")
        plt.ylabel("Frequency")
        plt.legend(loc = "upper right", bbox_to_anchor = (1.25, 1))
        plt.show()

        return embedding_model

#%%

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()

    processed_binary_df = data_preprocessor.preprocess_data(
        df = "3-back data/training_3back_data_wo_lures.csv",
        mode = "binary",
        output_path = "3-back data/test_3back_data_wo_lures.csv"
    )

    processed_multiclass_df = data_preprocessor.preprocess_data(
        df = "3-back data/training_3back_data_w_lures.csv",
        mode = "multiclass",
        output_path = "3-back data/test_3back_data_w_lures.csv"
    )
     
    X_train, y_train, X_val, y_val, X_test, y_test = data_preprocessor.split_data_for_bin_pred(processed_binary_df)
    X_test_with_lures, y_test_with_lures = data_preprocessor.split_data_for_mc_pred(processed_multiclass_df)

    print("Training dataset shape:", X_train.shape, y_train.shape)
    print("Validation dataset shape:", X_val.shape, y_val.shape)
    print("Test dataset shape:", X_test.shape, y_test.shape)
    print("X_test with lures shape", X_test_with_lures.shape)
    print("y_test with lures shape", y_test_with_lures.shape)

    rnn_trainer = RNNTrainer(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, X_test_w_lures = X_test_with_lures, y_test_w_lures = y_test_with_lures, n_batch = 128, learning_rate = 0.001)
    rnn_trainer.initialize_model()
    bce_history, model, training_history = rnn_trainer.train_model(epochs = 100)
    rnn_trainer.visualize_training_results()
    test_acc, pred_responses = rnn_trainer.eval_model_wo_lures(X_test, y_test, 3)
    rnn_trainer.visualize_preds_wo_lures(y_test, pred_responses, 3)
    test_acc_w_lures, pred_responses_w_lures = rnn_trainer.eval_model_w_lures(X_test_with_lures, y_test_with_lures, 3)
    rnn_trainer.visualize_preds_w_lures(y_test_with_lures, pred_responses_w_lures, 3)
    rnn_trainer.create_submodel(X_test_with_lures, y_test_with_lures, 3)
#%%