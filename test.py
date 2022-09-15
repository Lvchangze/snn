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

dataset = load_dataset("sst", split="train")
print(len(dataset))

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

# lstm = BiLSTM()
# x = torch.randn(32, 25, 100)
# o = lstm(x)
# print(o.shape)


# device_ids = [i for i in range(torch.cuda.device_count())]
# if torch.cuda.device_count() > 1:
#     print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
# if len(device_ids) > 1:
#     model = nn.DataParallel(model, device_ids=device_ids)

# glove_dict = {}
# with open("data/glove.6B.300d.txt", "r") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         glove_dict[word] = vector
# hidden_dim = glove_dict['the'].shape[-1]
# mean_value = np.mean(list(glove_dict.values()))
# variance_value = np.var(list(glove_dict.values()))
# left_boundary = mean_value - 3 * np.sqrt(variance_value)
# right_boundary = mean_value + 3 * np.sqrt(variance_value)
# for key in glove_dict.keys():
#     temp_clip = np.clip(glove_dict[key], left_boundary, right_boundary)
#     temp = (temp_clip - mean_value) / (3 * np.sqrt(variance_value))
#     glove_dict[key] = (temp + 1) / 2
# glove_dict = glove_dict
# glove_dict['<pad>'] = [0] * hidden_dim
# glove_dict['<unk>'] = [0] * hidden_dim
# sentence_length = 25

# def one_zero_normal(text_list, dict):
#     batch_embedding_list = []
#     for text in text_list:
#         text = text.lower()
#         text_embedding = []
#         words = list(map(lambda x: x if x in dict.keys() else '<unk>', text.strip().split()))
#         if len(words) > sentence_length:
#             words = words[:sentence_length]
#         elif len(words) < sentence_length:
#             while len(words) < sentence_length:
#                 words.append('<pad>')
#         for i in range(len(words)):
#             text_embedding.append(dict[words[i]])
#         batch_embedding_list.append(text_embedding)
#     return batch_embedding_list

# teacher_data_loader = DataLoader(dataset=TxtDataset(data_path="data/sst2/train_augment.txt"), batch_size= 50, shuffle=True)
# for i, batch in enumerate(teacher_data_loader):
#     # print(len(batch[0])) # 50
    
#     print(torch.tensor(one_zero_normal(list(batch[0]), glove_dict), dtype=float))
    
# with open("data/sst2/test_u_3v_sst2_glove300d_sent_len25.tensor_dataset", 'rb') as f:
#     test_dataset = pickle.load(f)
# print(len(test_dataset))

# with open("data/sst2/train_augment.txt") as fin:
#     lines = fin.readlines()
#     for line in lines:
#         print(line.strip())

