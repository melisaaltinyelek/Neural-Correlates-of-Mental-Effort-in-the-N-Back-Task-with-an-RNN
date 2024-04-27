#%%
import random

#%%

class NbackCreator():
    def __init__(self):
        global consonants
        consonants = "bcdfghjklmnprstvwxyz"
        self.consonants = consonants
        
        
    def create_sequences(self):
        self.cons_sequence = []

        for self.consonant in range(len(consonants)):
            self.consonant = random.choice(consonants)
            self.cons_sequence.append(self.consonant)
        
        return self.cons_sequence
    
#%%

n_back = NbackCreator()

sequence = n_back.create_sequences()

print(sequence)
# %%
