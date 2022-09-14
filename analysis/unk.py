import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from snntorch import spikegen
from model import SNN_TextCNN
from tqdm import tqdm
import re

def clean_tokenize(data, lower=False):
    ''' used to clean token, split all token with space and lower all tokens
    this function usually use in some language models which don't require strict pre-tokenization
    such as LSTM(with glove vector) or ELMO(already has tokenizer)
    :param data: string
    :return: list, contain all cleaned tokens from original input
    '''

    # recover some abbreviations
    data = re.sub(r"\-", " ", data)
    data = re.sub(r"\/", " ", data)
    data = re.sub(r"\s{2,}", " ", data)
    data = data.lower() if lower else data

    # split all tokens, form a list
    return [x.strip() for x in data.split() if x.strip()]



glove_dict = {}
with open("data/glove.6B.100d.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove_dict[word] = vector

sample_list = []
with open("data/sst2/train.txt", "r") as f:
    for line in f.readlines():
        temp = line.split('\t')
        sentence = temp[0].strip()
        label = int(temp[1])
        sample_list.append((sentence, label))


embedding_dim = 100
sent_length = 30

embedding_tuple_list = []
unk_num = 0
total_num = 0
for i in range(len(sample_list)):
    sent_embedding = np.array([[0] * embedding_dim] * sent_length, dtype=float)
    label = sample_list[i][1]
    text_list = sample_list[i][0].split()
    # text_list = clean_tokenize(sample_list[i][0])
    # print(sample_list[i][0], text_list)
    for j in range(len(text_list)):
        if text_list[j] not in glove_dict.keys():
            unk_num += 1
        total_num += 1
print(float(unk_num/total_num))