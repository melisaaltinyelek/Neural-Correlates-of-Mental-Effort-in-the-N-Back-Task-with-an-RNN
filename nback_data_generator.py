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
    
    experiment = NBackDataGen(N = 3, num_sequences = 150, num_minimum_trials = 57, letters = ["A", "B", "C", "D", "E", "F"])
    experiment.run_experiment()
#%%