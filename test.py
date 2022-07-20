import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from snntorch import spikegen
from model import TextCNN
# with open('data/train_u_3v_sst2_glove100d.tensor_dataset', 'rb') as f:
#     dataset = pickle.load(f)
# a = [[1,2,3],[4,6,5],[7,8,9],[10,11,12]]
# print(np.max(a))
# print(np.mean(a))
# print(np.var(a))
# b = [4,4,4]
# print(a[1] > b)
# print(spikegen.rate(torch.tensor([0], dtype=float), num_steps=20))

# a = np.array([1.0,2,3,4,5,6], dtype=float)
# print(a/6)
# print((np.array([1,2,3])-np.array([3,4,5]))/np.array([6,7,8]))
