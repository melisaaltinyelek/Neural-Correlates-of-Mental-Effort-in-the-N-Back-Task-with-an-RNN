#%%

import pandas as pd
import glob

from sweetpea import (
    Factor, DerivedLevel, WithinTrial, Transition, AtMostKInARow,
    MultiCrossBlock, CrossBlock, synthesize_trials, print_experiments,
    CMSGen, IterateGen, RandomGen, MinimumTrials, tabulate_experiments
)

from sweetpea import *

#%%

# PARAMETERS

N = 3 # N = 3 for 3-back task
num_sequences = 100 # Number of sequences
num_minimum_trials = 50 # Minimum number of trials per sequence

### REGULAR FACTORS

letters = ["A", "B", "C", "D", "E", "F"]

# Factor is used to define factors and its levels
curr_letter = Factor("letter", letters)

### DERIVED FACTORS

# Response factor

def is_target(letter):

    # If the letter N trials back matches the letter on the current trial, it's a target

    return letter[0] == letter[-N]

def is_lure(letter):
    
    return (letter[0] == letter[-(N-1)] or letter[0] == letter[-(N-2)]) and not is_target(letter)

def is_no_target(letter):   

    return not is_target(letter) and not is_lure(letter)


response = Factor("response", [
    DerivedLevel("target", Window(is_target, [curr_letter], 4, 1)),
    DerivedLevel("no target",  Window(is_no_target, [curr_letter], 4, 1)),
    DerivedLevel("is lure", Window(is_lure, [curr_letter], 4, 1))
])

# lure = Factor("lure", [
#     DerivedLevel("is lure", Window(is_lure, [curr_letter], 4, 1))
# ])

### EXPERIMENT

k = 7
constraints = [
               MinimumTrials(num_minimum_trials)]

design       = [curr_letter, response]
crossing     = [curr_letter, response]
block        = CrossBlock(design, crossing, constraints)

experiments  = synthesize_trials(block, num_sequences, CMSGen)

### END OF EXPERIMENT DESIGN

print_experiments(block, experiments)

tabulate_experiments(block, experiments, [curr_letter, response])

save_experiments_csv(block, experiments, "n_back_sequence")

# %%

# Store all csv files in a list for the current directory
csv_files = glob.glob("*.csv")

# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Loop through each csv file and append its contents to the dataframe
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    combined_df = pd.concat([combined_df.fillna("no target"), df], ignore_index = True)

combined_df.to_csv("data.csv", index = False)
# %%

