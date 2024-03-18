### added by Kevin
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
### end

from programs.rasp.sort import sort
from src.utils import data_utils


df =  data_utils.make_sort(vocab_size=10, dataset_size=10, min_length=5, max_length=7, seed=0)
print(df["sent"])
print(df["tags"])

a = sort.run(["<s>", "3", "1", "4", "2", "4", "0", "</s>"])

