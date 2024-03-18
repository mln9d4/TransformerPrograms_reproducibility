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
sort16 =  data_utils.make_sort(vocab_size=10, dataset_size=10000, min_length=16, max_length=18, seed=0)
print(sort16["sent"])
print(sort16["tags"])

# Create rest of datasets
reverse16 = data_utils.make_reverse(vocab_size=10, dataset_size=10000, min_length=16, max_length=18, seed=0)
histogram16 = data_utils.make_histogram(vocab_size=10, dataset_size=10000, min_length=16, max_length=18, seed=0)
double_histogram16 = data_utils.make_double_histogram(vocab_size=10, dataset_size=10000, min_length=16, max_length=18, seed=0)  
most_freq16 = data_utils.make_most_freq(vocab_size=10, dataset_size=10000, min_length=16, max_length=18, seed=0)    


a = sort.run(["<s>", "3", "1", "4", "2", "4", "0", "</s>"])


def create_dataset(num_sequences, sequence_length):
    dataset = []
    for _ in range(num_sequences):
        sequence = []
        for _ in range(sequence_length):
            sequence.append(str(random.randint(0, 9)))
        dataset.append(sequence)
    return dataset

dataset = create_dataset(2, 5)
print(f"Dataset: {dataset}")


ground_truth = []
for sequence in dataset:
    ground_truth.append(sorted(sequence))

ground_truth = sorted(ground_truth)
print(f"Ground truth: {ground_truth}")


# for sequence in dataset:
#     result = sort.run(["<s>"] + sequence + ["</s>"])
#     accuracy = 1.0 if result == ground_truth else 0.0
#     print(f"Sequence: {sequence}, Result: {result}, Accuracy: {accuracy}")

'''
for sequence in create_dataset(10, 5):
    result = sort.run(["<s>"] + sequence + ["</s>"])
    accuracy = 1.0 if result == ground_truth else 0.0
    print(f"Sequence: {sequence}, Result: {result}, Accuracy: {accuracy}")    
    for sequence in dataset:
        result = sort.run(["<s>"] + sequence + ["</s>"])
        accuracy = 1.0 if result == ground_truth else 0.0
        print(f"Sequence: {sequence}, Result: {result}, Accuracy: {accuracy}")
'''

# dataset = create_dataset(10, 5)
# print(f"Dataset: {dataset}")