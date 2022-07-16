import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

# with open('data/sst2_glove100d.tensor_dataset', 'rb') as f:
#     dataset = pickle.load(f)

# a = [[1,2,3],[4,6,5],[7,8,9],[10,11,12]]
# b = [4,4,4]
# print(a[1] > b)

from snnTorch.snntorch import surrogate
