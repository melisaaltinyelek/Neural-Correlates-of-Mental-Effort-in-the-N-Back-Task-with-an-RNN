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
    """
    A class to generate n-back task datasets using the SweetPea API.
    """

    def __init__(self, N, num_sequences, num_minimum_trials, letters):
        """
        Initializes the NBackDataGen class with n-back parameters.

        Parameters
        ----------
        N : int
            The n-back level for the task.
        num_sequences : int
            The number of sequences to generate.
        num_minimum_trials : int
            The minimum number of trials required in each sequence.
        letters : list of str
            A list of letters used in the task.
        """
        
        self.N = N
        self.num_sequences = num_sequences
        self.num_minimum_trials = num_minimum_trials
        self.letters = letters
        self.curr_letter = Factor("letter", letters)
        self.response = self.create_response_factor()
        self.constraints = [MinimumTrials(self.num_minimum_trials)]
        self.design = [self.curr_letter, self.response]
        self.crossing = [self.curr_letter, self.response]
        self.block = CrossBlock(self.design, self.crossing, self.constraints)

    def is_target(self, letter):
        """
        Determines whether a given letter sequence qualifies as a target trial.

        Parameters
        ----------
        letter : list of str
            A list of letters in the current sequence.

        Returns
        -------
        bool
            True if the first letter matches the letter n positions back, False otherwise.
        """
        
        return letter[0] == letter[-self.N]
    
    def is_lure(self, letter):
        """
        Determines whether a given letter sequence qualifies as a lure trial.

        Parameters
        ----------
        letter : list of str
            A list of letters in the current sequence.

        Returns
        -------
        bool
            True if the first letter matches an intermediate letter but not the target, False otherwise.
        """

        if self.N == 2:
            return (letter[0] == letter[-(self.N - 1)]) and not self.is_target(letter)
        elif self.N == 3:
            return (letter[0] == letter[-(self.N - 1)] or letter[0] == letter[-(self.N - 2)]) and not self.is_target(letter)
        elif self.N == 4:
            return (letter[0] == letter[-(self.N - 1)] or letter[0] == letter[-(self.N - 2)] or letter[0] == letter[-(self.N - 3)]) and not self.is_target(letter)
        elif self.N == 5:
            return (letter[0] == letter[-(self.N - 1)] or letter[0] == letter[-(self.N - 2)] or letter[0] == letter[-(self.N - 3)] or letter[0] == letter[-(self.N - 4)]) and not self.is_target(letter)
                        
    def is_no_target(self, letter):
        """
        Determines whether a given letter sequence qualifies as a non-target trial.

        Parameters
        ----------
        letter : list of str
            A list of letters in the current sequence.

        Returns
        -------
        bool
            True if the sequence is neither a target nor a lure, False otherwise.
        """
        
        return not self.is_target(letter) and not self.is_lure(letter)
        
    def create_response_factor(self):
        """
        Defines the response factor for the n-back task, categorizing trials as targets, non-targets, or lures.

        Returns
        -------
        Factor
            A SweetPea Factor object representing the response categories.
        """

        window_size = self.N + 1

        return Factor("response", [
            DerivedLevel("target", Window(self.is_target, [self.curr_letter], window_size, 1)),
            DerivedLevel("nontarget", Window(self.is_no_target, [self.curr_letter], window_size, 1)),
            DerivedLevel("lure", Window(self.is_lure, [self.curr_letter], window_size, 1))
        ])

    def gen_and_save_data(self):
        """
        Generates n-back task sequences and saves them as CSV files.
        """

        experiments = synthesize_trials(self.block, self.num_sequences, CMSGen)

        print_experiments(self.block, experiments)
        
        tabulate_experiments(self.block, experiments, [self.curr_letter, self.response])
        
        save_experiments_csv(self.block, experiments, "n_back_sequence")

        self.combine_csv_files()

    def combine_csv_files(self):
        """
        Combines multiple generated CSV files into a single dataset.
        """

        csv_files = glob.glob("*.csv")
        combined_df = pd.DataFrame()

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            combined_df = pd.concat([combined_df, df], ignore_index = True)
            os.remove(csv_file)  

        combined_df.dropna(inplace = True)
        combined_df.to_csv("raw_data.csv", index = False)
#%% 
if __name__ == "__main__":
    
    experiment = NBackDataGen(N = 5, num_sequences = 200, num_minimum_trials = 41, letters = ["A", "B", "C", "D", "E", "F"])
    experiment.gen_and_save_data()

    # num_minimum_trials = 38 counterbalances all letter - response pairs (2-back) 
    # num_minimum_trials = 39 counterbalances all letter - response pairs (3-back)
    # num_minimum_trials = 40 counterbalances all letter - response pairs (4-back)
    # num_minimum_trials = 41 counterbalances all letter - response pairs (5-back)
#%%