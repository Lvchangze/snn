from unicodedata import bidirectional
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# path = "/home/lvchangze/snn/saved_models/model_modeann-modedistill-student_model_namedpcnn-dataset_namesst2-label_num2-distill_loss_alpha0.3-distill_batch32-distill_epoch30/2022-09-17 14:01:18.log--epoch11.pth"
# saved_weights = torch.load(path)
# print(saved_weights.keys())


# with open("data/sst2/test_u_3v_sst2_glove100d_sent_len25.tensor_dataset", 'rb') as f:
#     dataset = pickle.load(f)
#     print(dataset[1])

# def get_samples_from_text(datafile_path):
#     sample_list = []
#     with open(datafile_path, "r") as f:
#         for line in f.readlines():
#             temp = line.split('\t')
#             sentence = temp[0].strip()
#             label = int(temp[1])
#             sample_list.append((sentence, label))
#     return sample_list
# dataset_name = "subj"
# samples = get_samples_from_text(f"data/{dataset_name}/all.txt")
# length = len(samples)
# random.shuffle(samples)
# train_samples = samples[:int(length * 0.85)]
# dev_samples = samples[int(length * 0.85):int(length * 0.9)]
# test_samples = samples[int(length * 0.9):]
# with open(f"data/{dataset_name}/train.txt", "w", encoding="utf-8")as f:
#     for sample in train_samples:
#         f.write(sample[0] + "\t" +str(sample[1])+"\n")
# with open(f"data/{dataset_name}/dev.txt", "w", encoding="utf-8")as f:
#     for sample in dev_samples:
#         f.write(sample[0] + "\t" +str(sample[1])+"\n")
# with open(f"data/{dataset_name}/test.txt", "w", encoding="utf-8")as f:
#     for sample in test_samples:
#         f.write(sample[0] + "\t" +str(sample[1])+"\n")

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)
# print(input)
# print(target)
# loss = F.cross_entropy(input, target)
# print(loss)
# input = torch.tensor([[50.0, 100.0],[64.0, 7.53]])
# output = torch.tensor([0, 0])
# print(F.cross_entropy(input, output))
# log_softmax_fn = nn.LogSoftmax(dim=-1)
# loss_fn = nn.NLLLoss()
# print(log_softmax_fn(input))
# print(loss_fn(log_softmax_fn(input), output))
print(torch.zeros(3,5))