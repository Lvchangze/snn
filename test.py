import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from snntorch import spikegen
from model import TextCNN
from tqdm import tqdm
import re
import math
# zero_index_in_this_batch = np.arange(301)
# print(np.intersect1d(zero_index_in_this_batch, [0,1,3]))

bias = 0.25

w = torch.empty(1, 100)
torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
w = w + bias
c = np.sum(w.cpu().detach().numpy() > 0)
print(float(c/(1*100)))

# torch.nn.init.xavier_normal_(w)
# w = w + bias
# c = np.sum(w.cpu().detach().numpy() > 0)
# print(float(c/(300*2)))
