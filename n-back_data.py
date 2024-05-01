#%%

import random
from random import randrange
import numpy as np

#%%

class NbackCreator():
    def __init__(self):

        """
        Stores all consonants in a list.
        """

        self.consonants = ["B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "R", "S", "T", "V", "W", "X", "Y", "Z"]

    def is_match(self, num_sequences):

        """
        Given a consonant list, picks a random letter from the random consonant_copy list and duplicates it.
        Appends that duplicate to the list and places it based on the 3-back rule.
        """

        match_sequences = []

        for _ in range(num_sequences):
            match_consonants_copy = self.consonants.copy()
            random.shuffle(match_consonants_copy)
            chosen_letter = random.choice(match_consonants_copy)
            idx_chosen_letter = match_consonants_copy.index(chosen_letter)
            match_consonants_copy.insert(idx_chosen_letter + 3, chosen_letter)
            match_sequences.append((idx_chosen_letter, match_consonants_copy)) 

        return match_sequences
    

    def is_mismatch(self, num_sequences):

        """
        Given a consonant list, picks a random letter from the random consonant_copy list and duplicates it.
        Appends that duplicate to the list and places it without satisfying the 3-back rule.
        The duplicated letter is randomly placed in the list.
        """

        mismatch_sequences = []
        
        for _ in range(num_sequences):
            mismatch_consonants_copy = self.consonants.copy()
            random.shuffle(mismatch_consonants_copy)
            chosen_letter = random.choice(mismatch_consonants_copy)
            idx_chosen_letter = mismatch_consonants_copy.index(chosen_letter)
            mismatch_consonants_copy.insert(randrange(len(mismatch_consonants_copy)), chosen_letter)
            mismatch_sequences.append((idx_chosen_letter, mismatch_consonants_copy))
        
        return mismatch_sequences

    
#%%

if __name__ == "__main__":

    n_back = NbackCreator()
    num_sequences = 22
     
    match_sequences = n_back.is_match(num_sequences)
    mismatch_sequences = n_back.is_mismatch(num_sequences)

    for idx, sequence in enumerate(match_sequences):
        print(sequence)
        
    for idx_2, sequence_2 in enumerate(mismatch_sequences):
        print(sequence_2)

# %%
