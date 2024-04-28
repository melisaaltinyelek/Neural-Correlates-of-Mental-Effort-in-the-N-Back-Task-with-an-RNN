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

        #self.consonants.append(self.chosen_letter)

        # Access the index of the chosen letter in the original list
        idx_chosen_letter = self.consonants.index(self.chosen_letter)
        
        # Access the last index of the list which represent the duplicate letter
        #idx_duplicate_letter = len(self.consonants) - 1

        # Insert the chosen letter (its duplicaste) 3 indices after its first occurence
        self.consonants.insert(idx_chosen_letter + 3, self.chosen_letter)

        return idx_chosen_letter, self.consonants

        

    def is_match(self, sequence):
        None

    def generate_trials(self):
        None

    
#%%

if __name__ == "__main__":
    
    n_back = NbackCreator()

    for _ in range(len(n_back.consonants)):
        
        sequence = n_back.generate_sequence()

        print(sequence)
# %%
