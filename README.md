# Studying Neural Correlates of Mental Effort in the Cognitive N-Back Task Using a Recurrent Neural Network

## 📌 Overview

The research aims to model working memory (WM) processes using a simple recurrent neural network (RNN) to simulate the n-back task and investigate cognitive mechanisms such as:

- **Generalization of the RNN model across n-back levels**  
- **Performance changes with increasing n-back levels**
- **Lure and trial-type-specific performance patterns**  
- **Neural and computational interpretations of task performance**

---

## 📂 Repository Structure  

    📁 n-back/                          # Main project directory

        ├── 📁 2-back data/             # Datasets for 2-back trials

        ├── 📁 3-back data/             # Datasets for 3-back trials

        ├── 📁 4-back data/             # Datasets for 4-back trials

        ├── 📁 5-back data/             # Datasets for 5-back trials

        ├── 📁 saved_model/             # Directory containing trained RNN model

        ├── 📄 .gitignore               # Files to ignore

        ├── 📄 README.md                # Documentation and project description

        ├── 📄 RNN_model.py             # Base RNN model implementation

        ├── 📄 analysis_2back.py        # Analysis script for 2-back condition

        ├── 📄 analysis_3back.py        # Analysis script for 3-back condition

        ├── 📄 analysis_4back.py        # Analysis script for 4-back condition

        ├── 📄 analysis_5back.py        # Analysis script for 5-back condition

        ├── 📄 nback_accuracies.json    # Stores RNN accuracy metrics

        ├── 📄 nback_data_gen.py        # Script to generate synthetic n-back task data

        ├── 📄 requirements.txt         # Dependencies required for running the project

        ├── 📄 save_and_plot_accuracies.py      # Script for saving and plotting RNN accuracies

---

## 🛠 Installation & Dependencies  
In order to set up the project, please install the required dependencies:

```bash
git clone <repo-link>
cd n-back
pip install -r requirements.txt