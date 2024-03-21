### added by Kevin
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
### end

from programs.rasp.sort import sort
from programs.rasp.reverse import reverse
from programs.rasp.hist import hist
from programs.rasp.double_hist import double_hist
from programs.rasp.most_freq import most_freq
from src.utils import data_utils
from output.rasp.sort.vocab8maxlen12.transformer_program.headsc4headsn4nlayers3cmlps2nmlps2.s0 import sort as sort8
from output.rasp.sort.vocab8maxlen8dvar100.transformer_program.headsc4headsn4nlayers3cmlps2nmlps2.s0 import sort as length_sort

# For the tasks reverse, histogram, double histogram, sort and most-freq the standard length of 8 sequences
# We try sequences of 16, 32 and 64

# Create dataset for sort
c=20
sort16 =  data_utils.make_sort(vocab_size=8, dataset_size=12, min_length=1, max_length=c, seed=0)

# Create rest of datasets
#reverse16 = data_utils.make_reverse(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)
#histogram16 = data_utils.make_hist(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)
#double_histogram16 = data_utils.make_double_hist(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)  
#most_freq16 = data_utils.make_most_freq(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)    

#print(sort16["sent"])
#print(len(sort16["sent"][0]))
#print(sort.run(sort16["sent"][0]))
#print(sort8.run(sort16["sent"][0]))
#a = sort.run(["<s>", "3", "1", "4", "2", "4", "0", "2", "4", "</s>"])
#print(a)
#def test_program(df, program):
#    for i in range(len(df)):
#        print(df["sent"][i])
#        print(df["tags"][i])
#        print(program.run(df["sent"][i]))

#test_program(sort16, sort)

def test_program(program):
    sequence_lengths = np.arange(10, 20, 10)
    sequence_lengths = np.array([6])
    results = {}
    for c in sequence_lengths:
        print(f"Sequence length: {c}")
        df =  data_utils.make_sort(vocab_size=8, dataset_size=12, min_length=c, max_length=c+2, seed=0)
        same = []
        for i in range(len(df)):
            print(f'sent: {df["sent"][i][1:-1]}')
            print(f'found: {program.run(df["sent"][i])[1:-1]}')
            print(f'tags: {df["tags"][i][1:-1]}')
            same.append(program.run(df["sent"][i])[1:-1] == df["tags"][i][1:-1])
        results[c] = same.count(True)/len(same)
    return results

results = test_program(length_sort)
print(results)

