#%%
import matplotlib.pyplot as plt
import pandas as pd
import json
#%%
def save_acc_to_json(task_name, accuracy_wo_lures, accuracy_w_lures, file_name = "nback_accuracies.json"):
    """
    Saves accuracy results for binary and multiclass classification tasks into a JSON file.

    Parameters
    ----------
    task_name : str
        The name of the task (e.g., "3-back", "4-back").
    accuracy_wo_lures : float
        The model accuracy for trials without lures.
    accuracy_w_lures : float
        The model accuracy for trials with lures.
    file_name : str, optional
        The JSON file where the results are stored.
    """

    results = {
        "accuracy_wo_lures": accuracy_wo_lures,
        "accuracy_w_lures": accuracy_w_lures
        }
    
    try:
        with open(file_name, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data[task_name] = results

    with open(file_name, "w") as file:
        json.dump(data, file, indent = 4)

def extract_accuracies(file_name = "nback_accuracies.json"):
    """
    Extracts accuracy results from the JSON file for different cognitive load levels.

    Parameters
    ----------
    file_name : str
        The JSON file containing accuracy results.

    Returns
    -------
    load_factors : list
        A list of cognitive load factors (e.g., "2-back", "3-back").
    accuracies_wo_lures : list
        A list of model accuracies for trials without lures.
    accuracies_w_lures : list
        A list of model accuracies for trials with lures.
    """

    with open(file_name, "r") as file:
        data = json.load(file)
        print(data)

    load_factors = []
    accuracies_wo_lures = []
    accuracies_w_lures = []

    for key, value in data.items():
        load_factors.append(key)

        if isinstance(value, dict):
            for k, v in value.items():
                if k == "accuracy_wo_lures":
                    accuracies_wo_lures.append(v)
                else:
                    accuracies_w_lures.append(v)
    
    print(f"Cognitive load factors: {load_factors}")
    print(f"Accuracy list without lures: {accuracies_wo_lures}")
    print(f"Accuracy list with lures: {accuracies_w_lures}")

    return load_factors, accuracies_wo_lures, accuracies_w_lures

def plot_accuracies(load_factors, accuracies_wo_lures, accuracies_w_lures):
    """
    Plots model accuracy for binary (no lures) and multiclass (with lures) n-back classifications.

    Parameters
    ----------
    load_factors : list
        A list of cognitive load levels (e.g., "2-back", "3-back").
    accuracies_wo_lures : list
        A list of model accuracies for trials without lures.
    accuracies_w_lures : list
        A list of model accuracies for trials with lures.
    """

    accuracies_wo_lures = map(float, accuracies_wo_lures)
    accuracies_w_lures = map(float, accuracies_w_lures)
    
    df = pd.DataFrame({"without lures": accuracies_wo_lures,
                       "with lures": accuracies_w_lures}, index = load_factors)
    
    ax = df.plot.bar(rot = 0, color = {"without lures": "#22CE83", "with lures": "#9172EC"})
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha = 'center', va = 'bottom', fontsize = 8)

    ax.set_title("Model Accuracy in Binary vs. Multiclass N-Back Classifications: Lures vs. No Lures")
    ax.yaxis.grid(True, linestyle = "--", linewidth = 0.5, color = "gray", alpha = 0.5)
    ax.set_xlabel("Cognitive Load Levels")
    ax.set_ylabel("Model Accuracy")
    ax.legend(loc = "upper right", bbox_to_anchor = (1.35, 1))
#%%
if __name__ == "__main__":

    load_factors, accuracies_wo_lures, accuracies_w_lures = extract_accuracies()

    plot_accuracies(load_factors, accuracies_wo_lures, accuracies_w_lures)
#%%
