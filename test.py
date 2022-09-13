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

# model  = TextCNN()
# model = model.load_state_dict(torch.load("saved_models/conversion.pth"))
# path = "/home/lvchangze/snn/saved_models/model_modeann-modetrain-dataset_namesst2-sentence_length25-dropout_p0.5-weight_decay0.001-batch_size32-learning_rate0.0005/2022-09-13 16:25:08.log--epoch29.pth"
# print(torch.load(path))

# with open("data/sst2/test_u_3v_sst2_glove100d_sent_len25.tensor_dataset", 'rb') as f:
#     dataset = pickle.load(f)
#     print(dataset[1])


def clean_tokenize(data, lower=False):
    # recover some abbreviations
    data = re.sub(r"\-", " ", data)
    data = re.sub(r"\/", " ", data)
    data = re.sub(r"(\s\.){2,}", " ", data)
    data = re.sub(r"\s{2,}", " ", data)
    data = data.lower() if lower else data

    # split all tokens, form a list
    return [x.strip() for x in data.split() if x.strip()]
print(clean_tokenize("' . . . mafia , rap stars and hood rats butt their ugly heads in a regurgitation of cinematic violence that gives brutal birth to an unlikely , but likable , hero . '"))