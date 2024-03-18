### added by Kevin
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
### end

from programs.rasp.sort import sort
from programs.rasp.reverse import reverse
from programs.rasp.hist import hist
from programs.rasp.double_hist import double_hist
from programs.rasp.most_freq import most_freq
from src.utils import data_utils

# For the tasks reverse, histogram, double histogram, sort and most-freq the standard length of 8 sequences
# We try sequences of 16, 32 and 64

# Create dataset for sort
b = 8
c = b+2
sort16 =  data_utils.make_sort(vocab_size=c, dataset_size=10000, min_length=b, max_length=c, seed=0)

# Create rest of datasets
reverse16 = data_utils.make_reverse(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)
histogram16 = data_utils.make_hist(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)
double_histogram16 = data_utils.make_double_hist(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)  
most_freq16 = data_utils.make_most_freq(vocab_size=8, dataset_size=10000, min_length=16, max_length=18, seed=0)    
print(sort16["sent"][0])
print(sort.run(sort16["sent"][0]))

a = sort.run(["<s>", "3", "1", "4", "2", "4", "0", "</s>"])

def test_program(df, program):
    for i in range(len(df)):
        print(df["sent"][i])
        print(df["tags"][i])
        print(program.run(df["sent"][i]))

#test_program(sort16, sort)

