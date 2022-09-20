import numpy as np
import re
import torch

class EmbeddingEncoder():
    def __init__(self, vocab_path, data_path, max_len=25, bias=3, need_norm = True) -> None:
        self.vocab_path = vocab_path
        self.bias = bias
        self.data_path = data_path
        self.max_len = max_len
        self.glove_dict = {}
        self.need_norm = need_norm
        self.get_embedding()
        pass

    def get_embedding(self):
        glove_dict = {}
        with open(self.vocab_path, "r") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        self.hidden_dim = glove_dict['the'].shape[-1]
        # print(len(glove_dict))
        # print(glove_dict.keys())
        # print(glove_dict['recent'])
        mean_value = np.mean(list(glove_dict.values()))
        vairance_value = np.var(list(glove_dict.values()))
        left = mean_value - np.sqrt(vairance_value) * self.bias
        right = mean_value + np.sqrt(vairance_value) * self.bias
        if self.need_norm:
            for key in glove_dict.keys():
                temp_clip = np.clip(glove_dict[key], left, right)
                temp = (temp_clip - mean_value) / (self.bias * np.sqrt(vairance_value))
                glove_dict[key] = (temp + 1) / 2
        self.glove_dict = glove_dict
        self.glove_dict['<pad>'] = [0] * self.hidden_dim
        self.glove_dict['<unk>'] = [0] * self.hidden_dim
        self.word2id = {}
        self.id2word = {}
        embedding_list = []
        for i, w in enumerate(self.glove_dict):
            self.word2id[w] = i
            self.id2word[i] = w
            embedding_list.append(torch.tensor(self.glove_dict[w]))
        self.embeddings = torch.stack(embedding_list, dim=0)

    def tokenize_sentence(self, sentence, lower=True):
        sentence = sentence.lower() if lower else sentence
        words = list(map(lambda x: x if x in self.glove_dict.keys() else '<unk>', sentence.strip().split()))
        if len(words) > self.max_len:
            words = words[:self.max_len]
        elif len(words) < self.max_len:
            while len(words) < self.max_len:
                words.append('<pad>')
        return [self.word2id[word] for word in words]

    def convert_ids_to_tokens(self, token_ids):
        return [self.id2word[id] for id in token_ids]

    def __call__(self, text_list):
        ids = []
        for item in text_list:
            ids.append(self.tokenize_sentence(item))
        # ids = [self.word2id[word] for word in words]
        embeddings = []
        for id in ids:
            embeddings.append(self.embeddings[id])
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings

    def tokenize(self, text_list):
        words = list(map(lambda x: x if x in self.glove_dict.keys() else '<unk>', text_list))
        ids = [self.word2id(word) for word in words]
        return self.embeddings[ids]

    @staticmethod
    def dataset_encode(data_path):
        sample_list = []
        with open(data_path, "r") as f:
            for line in f.readlines():
                temp = line.split('\t')
                sentence = temp[0].strip()
                label = int(temp[1])
                sample_list.append((sentence, label))
        return sample_list

    def encode(self):
        sample_list = []
        with open(self.data_path, "r") as f:
            for line in f.readlines():
                temp = line.split('\t')
                sentence = temp[0].strip()
                label = int(temp[1])
                sample_list.append((sentence, label))

    


if __name__ == '__main__':
    e = EmbeddingEncoder("data/glove.6B.300d.txt")
    e.get_embedding()