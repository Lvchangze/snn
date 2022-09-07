import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from model.textcnn import TextCNN
from snntorch import spikegen
from tqdm import tqdm
import re
import math
from datasets import load_dataset
# bias = 0.08

# w = torch.empty(300, 2)
# torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
# w = w + bias
# c = np.sum(w.cpu().detach().numpy() > 0)
# print(float(c/(300*2)))

# w = torch.ones(20, 32, 100)
# print(w.size())
# print(w[0,:,0:50].sum(-1).size())
# print(w[0,:,0:50].sum(-1))
# print(w[0,:,0:50].sum(-1).sum(0).size())
# print(w[0,:,0:50].sum(-1).sum(0))
# rate = spikegen.rate(w, num_steps=30)
# print(rate.size())
# latency = spikegen.latency(w, num_steps=30)
# print(latency.size())

# dataset = load_dataset("sst2", split="test")
# print(dataset[0])
model  = TextCNN()
model = model.load_state_dict(torch.load("saved_models/conversion.pth"))