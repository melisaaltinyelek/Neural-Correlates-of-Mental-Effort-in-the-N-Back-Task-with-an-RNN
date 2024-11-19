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

class NBackDataGen:
    def __init__(self, N, num_sequences, num_minimum_trials, letters):
        
        self.N = N
        self.num_sequences = num_sequences
        self.num_minimum_trials = num_minimum_trials
        self.letters = letters
        self.curr_letter = Factor("letter", letters)
        self.response = self.create_response_factor()
        self.constraints = [MinimumTrials(self.num_minimum_trials),
                            AtMostKInARow(3, self.response)]
        self.design = [self.curr_letter, self.response]
        self.crossing = [self.curr_letter, self.response]
        self.block = CrossBlock(self.design, self.crossing, self.constraints)

    def is_target(self, letter):
        
        return letter[0] == letter[-self.N]

    def is_lure(self, letter):
        
        return (letter[0] == letter[-(self.N - 1)] or letter[0] == letter[-(self.N - 2)]) and not self.is_target(letter)

    def is_no_target(self, letter):
        
        return not self.is_target(letter) and not self.is_lure(letter)

    def create_response_factor(self):
        
        return Factor("response", [
            DerivedLevel("target", Window(self.is_target, [self.curr_letter], 4, 1)),
            DerivedLevel("nontarget", Window(self.is_no_target, [self.curr_letter], 4, 1)),
            DerivedLevel("lure", Window(self.is_lure, [self.curr_letter], 4, 1))
        ])

    def run_experiment(self):
        
        experiments = synthesize_trials(self.block, self.num_sequences, CMSGen)

        print_experiments(self.block, experiments)
        
        tabulate_experiments(self.block, experiments, [self.curr_letter, self.response])
        
        save_experiments_csv(self.block, experiments, "n_back_sequence")

        self.combine_csv_files()

    def combine_csv_files(self):

        csv_files = glob.glob("*.csv")
        combined_df = pd.DataFrame()

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            combined_df = pd.concat([combined_df, df], ignore_index = True)
            os.remove(csv_file)  

        combined_df.dropna(inplace = True)
        combined_df.to_csv("raw_data.csv", index = False)
    
if __name__ == "__main__":
    
    experiment = NBackDataGen(N = 3, num_sequences = 3, num_minimum_trials = 57, letters = ["A", "B", "C", "D", "E", "F"])
    experiment.run_experiment()

#%%

# N = 3 # N = 3 for 3-back task
# num_sequences = 150 # Number of sequences
# num_minimum_trials = 57 # Minimum number of trials per sequence

# letters = ["A", "B", "C", "D", "E", "F"]

# curr_letter = Factor("letter", letters)

# def is_target(letter):

#     # If the letter N trials back matches the letter on the current trial, it's a target
#     return letter[0] == letter[-N]

# def is_lure(letter):
    
#     return (letter[0] == letter[-(N-1)] or letter[0] == letter[-(N-2)]) and not is_target(letter)
#     # return not is_target(letter)

# def is_no_target(letter):   

#     return not is_target(letter) and not is_lure(letter)

# response = Factor("response", [
#     DerivedLevel("target", Window(is_target, [curr_letter], 4, 1)),
#     DerivedLevel("nontarget", Window(is_no_target, [curr_letter], 4, 1)),
#     DerivedLevel("lure", Window(is_lure, [curr_letter], 4, 1))
# ])

# num_letters = len(letters)
# #print(f"The total number of letters: {num_letters}")

# num_responses = len(response.levels)
# #print(f"The total number of responses: {num_responses}")

# total_combinations = num_letters * num_responses
# trials_per_combination = num_minimum_trials // total_combinations

# # trials_per_letter = num_minimum_trials // num_letters
# # print(f"Trials per letter: {trials_per_letter}")

# # trials_per_response = num_minimum_trials // num_responses
# # print(f"Trials per response: {trials_per_response}")

# num_consecutive_response = 3
# constraints = [MinimumTrials(num_minimum_trials),
#                AtMostKInARow(num_consecutive_response, response)
#                ]

# design       = [curr_letter, response]
# crossing     = [curr_letter, response]
# block        = CrossBlock(design, crossing, constraints)

# experiments  = synthesize_trials(block, num_sequences, CMSGen)

# print_experiments(block, experiments)

# tabulate_experiments(block, experiments, [curr_letter, response])

# save_experiments_csv(block, experiments, "n_back_sequence")

# # Store all csv files in a list for the current directory
# csv_files = glob.glob("*.csv")

# # Create an empty dataframe to store the combined data
# combined_df = pd.DataFrame()

# # Loop through each csv file and append its contents to the dataframe
# for csv_file in csv_files:
#     df = pd.read_csv(csv_file)
#     combined_df = pd.concat([combined_df, df], ignore_index = True)

#     os.remove(csv_file)

# combined_df.dropna(inplace = True)

# combined_df.to_csv("raw_data_with_lure.csv", index = False)