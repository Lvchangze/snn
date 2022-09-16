from unicodedata import bidirectional
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.textcnn import SNN_TextCNN
from snntorch import spikegen
from tqdm import tqdm
import re
import math
from datasets import load_dataset
import random
import nltk
from nltk import word_tokenize
from dataset import TxtDataset
from transformers import BertTokenizer
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

# dataset = load_dataset("sst", split="train")
# print(len(dataset))

# model  = TextCNN()
# model = model.load_state_dict(torch.load("saved_models/conversion.pth"))
# path = "/home/lvchangze/snn/saved_models/model_modeann-modetrain-dataset_namesst2-sentence_length25-dropout_p0.5-weight_decay0.001-batch_size32-learning_rate0.0005/2022-09-13 16:25:08.log--epoch29.pth"
# print(torch.load(path))

# with open("data/sst2/test_u_3v_sst2_glove100d_sent_len25.tensor_dataset", 'rb') as f:
#     dataset = pickle.load(f)
#     print(dataset[1])


# class BiLSTM(nn.Module):
#     def __init__(self) -> None:
#         super(BiLSTM, self).__init__()
#         self.lstm = nn.LSTM(batch_first=True, input_size=100, hidden_size=150, num_layers=1, bidirectional=True, bias=False)
#         self.fc_1 = nn.Linear(150 * 2, 200, bias=False)
#         self.relu = nn.ReLU()
#         self.output_fc = nn.Linear(200, 2, bias=False)

#     def forward(self, x):
#         print(x.shape)
#         output, (hidden,cell) = self.lstm(x)
#         x = self.fc_1(output)
#         x = self.relu(x)
#         fc_output = self.output_fc(x)
#         fc_output = fc_output[:,-1,:].squeeze(1)
#         return fc_output
lrn = nn.LocalResponseNorm(2)
signal_2d = torch.randn(32, 5, 24, 24)
signal_4d = torch.randn(16, 5, 7, 7, 7, 7)



# device_ids = [i for i in range(torch.cuda.device_count())]
# if torch.cuda.device_count() > 1:
#     print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
# if len(device_ids) > 1:
#     model = nn.DataParallel(model, device_ids=device_ids)

