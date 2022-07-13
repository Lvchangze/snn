import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

with open('data/sst2_glove100d.tensor_dataset', 'rb') as f:
    dataset = pickle.load(f)

print(dataset[0])