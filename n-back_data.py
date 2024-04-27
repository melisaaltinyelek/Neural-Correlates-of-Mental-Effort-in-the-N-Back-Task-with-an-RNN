#%%
import random

#%%

class NbackCreator():
    def __init__(self):

        """
        Stores all consonants in a list.
        """

        self.consonants = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "x", "y", "z"]

    def generate_sequence(self):

        """
        Given a consonant list, picks a random letter and duplicates it.
        Appends that duplicate to the list.
        """

        random.shuffle(self.consonants)
        self.chosen_letter = random.choice(self.consonants)

        self.consonants.append(self.chosen_letter)


        return self.consonants


    def is_match(self, sequence):
        None

    def generate_trials(self):
        None

    
#%%

n_back = NbackCreator()

sequence = n_back.generate_sequence()

print(sequence)
# %%
