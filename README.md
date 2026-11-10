# Neural Correlates of Mental Effort in the N-Back Task Using an RNN

**Master's Thesis Project | Cognitive Science – Artificial Intelligence & Neuroscience | Universität Osnabrück**

**Tech Stack:** Python · TensorFlow · Keras · NumPy · Pandas · scikit-learn · Matplotlib · SweetPea

## Overview

This project investigates working memory and cognitive effort through computational modeling of the **N-Back task**, using a Recurrent Neural Network (RNN).

The N-Back task is commonly used in cognitive neuroscience to study working memory. Participants must determine whether a current stimulus matches one presented *N* steps earlier in a sequence. As *N* increases, the task generally becomes more cognitively demanding.

The project implements and evaluates an RNN across multiple N-Back conditions to investigate how task complexity and trial characteristics relate to model performance.

## Research Objectives

The project explores four main questions:

1. **Generalization:** How does an RNN trained to perform the N-Back task generalize across different levels of task difficulty?
2. **Cognitive Load:** How does model performance change as the N-Back level increases?
3. **Trial-Specific Performance:** How does the network respond to target, non-target, and lure trials?
4. **Computational Interpretation:** What can the network's behavior reveal about working memory and cognitive processing?

## Model Architecture

The neural network is implemented using TensorFlow and Keras.

The architecture consists of:

- A `SimpleRNN` layer for processing sequential inputs.
- A fully connected hidden layer with `tanh` activation.
- A sigmoid output layer for binary classification.

The implementation uses the **Adam optimizer**, **binary cross-entropy loss**, and accuracy as a training metric.

The model is trained using generated N-Back task data, with separate validation and evaluation procedures.

## Methodology

### 1. Data Generation

Synthetic N-Back datasets are generated programmatically for different levels of task difficulty.

The experiments cover:

- 2-Back
- 3-Back
- 4-Back
- 5-Back

The data-generation process supports controlled experimental conditions and the investigation of different trial types.

### 2. Model Training

The RNN is trained to learn the relationship between input sequences and their corresponding task responses.

The training implementation includes:

- Training and validation datasets
- Configurable training parameters
- Model checkpoint saving
- Tracking training and validation loss
- Tracking training and validation accuracy

### 3. Model Evaluation

The trained model is evaluated across different N-Back conditions, with particular attention to performance differences between trial types.

The evaluation code includes:

- Overall classification accuracy
- Target and non-target accuracy
- Lure-related performance analysis
- Confusion matrices
- Training and validation visualizations

These analyses help characterize the network's performance under different task conditions.

## Repository Structure

| File / Directory | Description |
|---|---|
| `RNN_model.py` | Neural network implementation, training, and evaluation |
| `nback_data_gen.py` | Synthetic N-Back task data generation |
| `analysis_2back.py` | Evaluation of the 2-Back condition |
| `analysis_3back.py` | Evaluation of the 3-Back condition |
| `analysis_4back.py` | Evaluation of the 4-Back condition |
| `analysis_5back.py` | Evaluation of the 5-Back condition |
| `save_and_plot_accuracies.py` | Accuracy aggregation and visualization |
| `nback_accuracies.json` | Stored model accuracy results |
| `saved_model/` | Saved neural network model |
| `2-back data/` – `5-back data/` | Experimental datasets |
| `requirements.txt` | Python dependencies |

## Technologies

**Machine Learning & Deep Learning**
- TensorFlow and Keras
- Recurrent Neural Networks
- scikit-learn

**Data Processing & Analysis**
- Python
- NumPy
- Pandas
- SweetPea

**Visualization & Evaluation**
- Matplotlib
- Classification metrics
- Confusion matrices
- Accuracy and loss visualization

## Installation

Clone the repository:

`git clone https://github.com/melisaaltinyelek/Neural-Correlates-of-Mental-Effort-in-the-N-Back-Task-with-an-RNN.git`

Navigate to the project directory:

`cd Neural-Correlates-of-Mental-Effort-in-the-N-Back-Task-with-an-RNN`

Install dependencies:

`pip install -r requirements.txt`

The repository contains separate scripts for data generation, model implementation, and analysis of individual N-Back conditions. Review their configurations and dataset paths before running experiments.

## Research Scope

This project was developed as part of my Master's research in Cognitive Science, specializing in Artificial Intelligence and Neuroscience, at the Universität Osnabrück.

Its primary focus is the computational investigation of working memory through neural network modeling, rather than the development of a production-ready AI application.

The repository demonstrates practical experience in:

- Implementing and training neural networks using TensorFlow/Keras
- Generating and processing experimental datasets
- Designing computational experiments
- Evaluating machine learning model performance
- Analyzing and visualizing experimental results
- Connecting machine learning methods with cognitive science research

## Author

**Melisa Altinyelek**

M.Sc. Cognitive Science — Artificial Intelligence & Neuroscience  
Universität Osnabrück
