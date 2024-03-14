### added by Kevin
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
### end

from programs.rasp.sort import sort
import random


# sort.run(["<s>", "3", "1", "4", "2", "4", "0", "</s>"])

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