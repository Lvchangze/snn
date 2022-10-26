import sys
sys.path.append("..")
import re
import numpy as np
import pickle
import torch
from tqdm import tqdm
import os
from dataset import TensorDataset
from datasets import load_dataset
import nltk
from jieba import tokenize
import jieba
def get_samples_from_text(datafile_path):
    sample_list = []
    with open(datafile_path, "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sample_list.append((sentence, label))
    return sample_list


def get_samples_from_web(dataset_name, data_type):
    sample_list = []
    dataset = load_dataset(dataset_name, split=data_type)
    for sample in dataset:
        sentence = sample['sentence'].strip()
        label = int(sample['label'])
        sample_list.append((sentence, label))
    return sample_list

class TensorEncoder():
    def __init__(self, vocab_path, dataset_name, datafile_path, sent_length:int, embedding_dim:int, data_type="trian", bias=3) -> None:
        super(TensorEncoder, self).__init__()
        self.vocab_path = vocab_path
        self.dataset_name = dataset_name
        self.datafile_path = datafile_path
        self.sent_length = sent_length
        self.embedding_dim = embedding_dim
        self.data_type = data_type
        self.bias = bias

    def encode(self):
        glove_dict = {}
        with open(self.vocab_path, "r") as f:
            word_count=0
            for line in f:
                values = line.split()
                word = values[0]
                word_count+=1
                # print(word)
                # if (word_count>100):
                #     assert False
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector

        mean_embedding = np.mean(np.array(list(glove_dict.values())), axis=0)
        zero_embedding = np.array([0] * self.embedding_dim, dtype=float)
        mean_value = np.mean(list(glove_dict.values()))
        print(mean_value.shape)
        variance_value = np.var(list(glove_dict.values()))
        left_boundary = mean_value - self.bias * np.sqrt(variance_value)
        right_boundary = mean_value + self.bias * np.sqrt(variance_value)

        sample_list = get_samples_from_text(datafile_path=self.datafile_path)
        # sample_list = get_samples_from_web(self.dataset_name, self.data_type)

        embedding_tuple_list = []
        for i in tqdm(range(len(sample_list))):
            sent_embedding = np.array([[0] * self.embedding_dim] * self.sent_length, dtype=float)
            text_list = nltk.word_tokenize(sample_list[i][0])
            label = sample_list[i][1]
            for j in range(self.sent_length):
                if j >= len(text_list):
                    embedding_norm = zero_embedding # zero padding
                else:
                    word = text_list[j]
                    embedding = glove_dict[word] if word in glove_dict.keys() else zero_embedding
                    # N(0, 1)
                    embedding_n01 = (embedding - np.array([mean_value] * self.embedding_dim)) / np.array([np.sqrt(variance_value)] * self.embedding_dim)
                    embedding_norm = np.array([0] * self.embedding_dim, dtype=float)
                    for k in range(self.embedding_dim):
                        if embedding[k] < left_boundary:
                            embedding_norm[k] = -self.bias
                        elif embedding[k] > right_boundary:
                            embedding_norm[k] = self.bias
                        else:
                            embedding_norm[k] = embedding_n01[k]
                    # add abs(left_embedding)
                    embedding_norm = (embedding_norm + np.array([np.abs(self.bias)] * self.embedding_dim))/(self.bias * 2)
                    # embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
                sent_embedding[j] = embedding_norm
            # print(i, sent_embedding)
            embedding_tuple_list.append((torch.tensor(sent_embedding, dtype=float), label))
        
        dataset = TensorDataset(embedding_tuple_list)

        file_name = f"../data/{self.dataset_name}/{self.data_type}_u_{self.bias}v_{self.dataset_name}_glove{self.embedding_dim}d_sent_len{self.sent_length}.tensor_dataset"
        if not os.path.exists(file_name):
            with open(file_name, 'wb') as f:
                pickle.dump(dataset, f, -1)
                
        return dataset

class ChineseTensorEncoder():
    def __init__(self, vocab_path,vocab_dict_path, dataset_name, datafile_path, sent_length:int, embedding_dim:int, data_type="trian", bias=3) -> None:
        super(ChineseTensorEncoder, self).__init__()
        self.vocab_path = vocab_path
        self.vocab_dict_path = vocab_dict_path
        self.dataset_name = dataset_name
        self.datafile_path = datafile_path
        self.sent_length = sent_length
        self.embedding_dim = embedding_dim
        self.data_type = data_type
        self.bias = bias
    def process_vocab(self):
        word_embedding_dict = {}
        with open(self.vocab_path, "r") as f:
            word_count=0
            for line in f:
                word_count+=1
                if(word_count==1):
                    continue
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], "float32")
                word_embedding_dict[word] = vector

    def encode(self):
        word_embedding_dict = {}
        word_embedding_list=[]
        with open(self.vocab_path, "r") as f:
            word_count=0
            for line in f:
                word_count+=1
                if(word_count==1):
                    continue
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], "float32")
                word_embedding_dict[word] = vector
                word_embedding_list.append(vector.tolist())
        print(np.array(word_embedding_list).shape)
        mean_embedding = np.mean(np.array(list(word_embedding_dict.values())), axis=0)
        zero_embedding = np.array([0] * self.embedding_dim, dtype=float)
        mean_value = np.mean(np.array(word_embedding_list))
        variance_value = np.var(np.array(word_embedding_list))
        print(mean_value)
        print(variance_value)
        left_boundary = mean_value - self.bias * np.sqrt(variance_value)
        right_boundary = mean_value + self.bias * np.sqrt(variance_value)

        sample_list = get_samples_from_text(datafile_path=self.datafile_path)

        embedding_tuple_list = []
        for i in tqdm(range(len(sample_list))):
            sent_embedding = np.array([[0] * self.embedding_dim] * self.sent_length, dtype=float)
            text_list = list(jieba.tokenize(sample_list[i][0]))
            label = sample_list[i][1]
            for j in range(self.sent_length):
                if j >= len(text_list):
                    embedding_norm = zero_embedding # zero padding
                else:
                    word = text_list[j][0]
                    embedding = word_embedding_dict[word] if word in word_embedding_dict.keys() else zero_embedding
                    # N(0, 1)
                    embedding_n01 = (embedding - np.array([mean_value] * self.embedding_dim)) / np.array([np.sqrt(variance_value)] * self.embedding_dim)
                    embedding_norm = np.array([0] * self.embedding_dim, dtype=float)
                    for k in range(self.embedding_dim):
                        if embedding[k] < left_boundary:
                            embedding_norm[k] = -self.bias
                        elif embedding[k] > right_boundary:
                            embedding_norm[k] = self.bias
                        else:
                            embedding_norm[k] = embedding_n01[k]
                    # add abs(left_embedding)
                    embedding_norm = (embedding_norm + np.array([np.abs(self.bias)] * self.embedding_dim))/(self.bias * 2)
                    # embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
                sent_embedding[j] = embedding_norm
            # print(i, sent_embedding)
            embedding_tuple_list.append((torch.tensor(sent_embedding, dtype=float), label))
        
        dataset = TensorDataset(embedding_tuple_list)
        
        file_name = f"../data/{self.dataset_name}/{self.data_type}_u_{self.bias}v_{self.dataset_name}_word2vec{self.embedding_dim}d_sent_len{self.sent_length}.tensor_dataset"
        if not os.path.exists(file_name):
            with open(file_name, 'wb') as f:
                pickle.dump(dataset, f, -1)
                
        return dataset
if __name__ == "__main__":
    tensor_encoder = ChineseTensorEncoder(
        vocab_path="../word2vec/sgns.merge.bigram",
        vocab_dict_path="../word2vec/sgns.merge.bigram.pkl",
        dataset_name="senti",
        data_type="test",
        datafile_path="../dataset/txt/chnsenti_test.txt",
        sent_length=32,
        embedding_dim=300,
        bias = 3
    )
    tensor_encoder.encode()