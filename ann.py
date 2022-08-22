import pickle
import argparse
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re
from dataset import TensorDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--sent_length",
    default=30,
    type=int
)
parser.add_argument(
    "--batch_size",
    default=50,
    type=int
)
parser.add_argument(
    "--lr",
    default=5e-5,
    type=float
)
parser.add_argument(
    "--dropout_p",
    default=0.5,
    type=float
)
parser.add_argument(
    "--decay",
    default=0.01,
    type=float
)
parser.add_argument(
    "--embedding_dim",
    default=100,
    type=int
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int
)
args = parser.parse_args()

glove_dict = {}
with open(f"data/glove.6B.{args.embedding_dim}d.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove_dict[word] = vector
mean_embedding = np.mean(np.array(list(glove_dict.values())), axis=0)
zero_embedding = np.array([0] * args.embedding_dim, dtype=float)

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

class TextCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=args.embedding_dim, kernel_size=(filter_size, args.embedding_dim)) for filter_size in [3,4,5]
        ])
        self.leaky = nn.LeakyReLU()
        self.avgpool_1 = nn.ModuleList([
            nn.AvgPool2d((args.sent_length - filter_size + 1, 1)) for filter_size in [3,4,5]
        ])
        self.drop = nn.Dropout(p=args.dropout_p)
        self.fc = nn.Linear(len([3,4,5]) * args.embedding_dim, 2)
        
    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=1)
        conv_out = [conv(x) for conv in self.convs_1]
        leaky_out = [self.leaky(conv) for conv in conv_out]
        pooled_out = [self.avgpool_1[i](leaky_out[i]) for i in range(len(self.avgpool_1))]
        pooled_out = [self.drop(pool) for pool in pooled_out]
        flatten = torch.cat(pooled_out, dim=1).view(batch_size, -1)
        fc_output = self.fc(flatten)
        return fc_output

def get_tensor_dataset(file):
    sample_list = []
    with open(file, "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sample_list.append((sentence, label))

    embedding_tuple_list = []
    for i in range(len(sample_list)):
        sent_embedding = np.array([[0] * args.embedding_dim] * args.sent_length, dtype=float)
        # text_list = sample_list[i][0].split()
        text_list = clean_tokenize(sample_list[i][0])
        label = sample_list[i][1]
        for j in range(args.sent_length):
            if j >= len(text_list):
                embedding = zero_embedding # zero padding
            else:
                word = text_list[j]
                embedding = glove_dict[word] if word in glove_dict.keys() else zero_embedding
            sent_embedding[j] = embedding
        embedding_tuple_list.append((torch.tensor(sent_embedding), label))
    dataset = TensorDataset(embedding_tuple_list)
    return dataset


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TextCNN()
    net.to(device=device)
    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=args.decay, betas=(0.9, 0.999))

    # with open("data/sst2/train_u_3v_sst2_glove100d.tensor_dataset", "rb") as f:
    #     train_data = pickle.load(f)
    train_data = get_tensor_dataset("data/sst2/train.txt")
    train_dataloader = DataLoader(train_data,batch_size=args.batch_size, shuffle=True)

    # with open("data/sst2/test_u_3v_sst2_glove100d.tensor_dataset", "rb") as f:
    #     test_data = pickle.load(f)

    test_data = get_tensor_dataset("data/sst2/test.txt")
    all = len(test_data)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    acc_list = []
    for epoch in tqdm(range(args.epochs)):
        for data, target in train_dataloader:
            net.train()
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            correct = 0
            for data, y_batch in test_dataloader:
                data = data.to(device)
                y_batch = y_batch.to(device)
                output = net(data)
                correct += int(y_batch.eq(torch.max(output,1)[1]).sum())
        acc_list.append(float(correct/all))
        print(f"Epoch {epoch} Acc: {float(correct/all)}")
    print(np.max(acc_list))