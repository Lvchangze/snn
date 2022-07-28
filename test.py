import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from snntorch import spikegen
from model import TextCNN
from tqdm import tqdm
import re

# zero_index_in_this_batch = np.arange(301)
# print(np.intersect1d(zero_index_in_this_batch, [0,1,3]))

bias = 0.0364

w = torch.zeros(32, 300, dtype=float)
print(w)
torch.nn.init.kaiming_normal_(w)
w = w + bias
c = np.sum(w.cpu().detach().numpy() > 0)
print(float(c/(100*100)))
