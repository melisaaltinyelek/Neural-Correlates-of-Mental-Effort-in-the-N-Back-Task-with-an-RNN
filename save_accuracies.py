import json
import os

def save_acc_to_json(task_name, accuracy_wo_lures, accuracy_w_lures, file_name = "nback_accuracies.json"):

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