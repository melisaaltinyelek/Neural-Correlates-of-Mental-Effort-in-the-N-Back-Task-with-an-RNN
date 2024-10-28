#%%

import pandas as pd
import glob
import os

from sweetpea import (
    Factor, DerivedLevel, WithinTrial, Transition, AtMostKInARow, ExactlyK,
    MultiCrossBlock, CrossBlock, synthesize_trials, print_experiments,
    CMSGen, IterateGen, RandomGen, MinimumTrials, tabulate_experiments
)

from sweetpea import *
#%%

N = 3 # N = 3 for 3-back task
num_sequences = 150 # Number of sequences
num_minimum_trials = 57 # Minimum number of trials per sequence

letters = ["A", "B", "C", "D", "E", "F"]

# Factor is used to define factors and its levels
curr_letter = Factor("letter", letters)

def is_target(letter):

    # If the letter N trials back matches the letter on the current trial, it's a target

    return letter[0] == letter[-N]

def is_lure(letter):
    
    return (letter[0] == letter[-(N-1)] or letter[0] == letter[-(N-2)]) and not is_target(letter)
    #return not is_target(letter)

def is_no_target(letter):   

    return not is_target(letter) and not is_lure(letter)

response = Factor("response", [
    DerivedLevel("target", Window(is_target, [curr_letter], 4, 1)),
    DerivedLevel("nontarget", Window(is_no_target, [curr_letter], 4, 1)),
    DerivedLevel("lure", Window(is_lure, [curr_letter], 4, 1))
])

num_letters = len(letters)
#print(f"The total number of letters: {num_letters}")

num_responses = len(response.levels)
#print(f"The total number of responses: {num_responses}")

total_combinations = num_letters * num_responses
trials_per_combination = num_minimum_trials // total_combinations

# trials_per_letter = num_minimum_trials // num_letters
# print(f"Trials per letter: {trials_per_letter}")

# trials_per_response = num_minimum_trials // num_responses
# print(f"Trials per response: {trials_per_response}")

num_consecutive_response = 3
constraints = [MinimumTrials(num_minimum_trials),
               AtMostKInARow(num_consecutive_response, response)

                #ExactlyK(trials_per_combination, curr_letter),
                #ExactlyK(trials_per_combination, response)
               
                #AtMostKInARow(response_constraint, response),
                # ExactlyK(trials_per_response, response.levels[0]),
                # ExactlyK(trials_per_response, response.levels[1]),
                # ExactlyK(trials_per_response, response.levels[2]),
               ]

# constraints.extend([
#     ExactlyK(trials_per_letter, curr_letter.levels[0]),
#     ExactlyK(trials_per_letter, curr_letter.levels[1]),
#     ExactlyK(trials_per_letter, curr_letter.levels[2]),
#     ExactlyK(trials_per_letter, curr_letter.levels[3]),
#     ExactlyK(trials_per_letter, curr_letter.levels[4]),
#     ExactlyK(trials_per_letter, curr_letter.levels[5]),
# ])

design       = [curr_letter, response]
crossing     = [curr_letter, response]
block        = CrossBlock(design, crossing, constraints)

experiments  = synthesize_trials(block, num_sequences, CMSGen)

print_experiments(block, experiments)

tabulate_experiments(block, experiments, [curr_letter, response])

save_experiments_csv(block, experiments, "n_back_sequence")

# Store all csv files in a list for the current directory
csv_files = glob.glob("*.csv")

# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Loop through each csv file and append its contents to the dataframe
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    combined_df = pd.concat([combined_df, df], ignore_index = True)

    os.remove(csv_file)

combined_df.dropna(inplace = True)

combined_df.to_csv("raw_data_with_lure.csv", index = False)
#%%