import torch
import torch.nn as nn
import pickle
import argparse
from numpy import dtype
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re
import time
import random
import nltk
from nltk import word_tokenize

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        default="../data/sst2/train.txt",
        type=str,
    )
    parser.add_argument(
        "--test_data_path",
        default="../data/sst2/test.txt",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        default="../data/sst2/train_augment.txt",
        type=str,
    )
    parser.add_argument(
        "--n_iter",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--p_mask",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--p_pos",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--p_ng",
        default=0.25,
        type=float,
    )
    args = parser.parse_args()
    return args

def get_samples_from_text(datafile_path):
    sample_list = []
    with open(datafile_path, "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sample_list.append((sentence, label))
    return sample_list

def get_poc_tag_dict(datafile_path):
    pos_dict={}
    with open(datafile_path, "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            tuple_list = nltk.pos_tag(word_tokenize(sentence))
            for tuple in tuple_list:
                if tuple[1] not in pos_dict.keys():
                    pos_dict[tuple[1]] = []
                if tuple[0] not in pos_dict[tuple[1]]:
                    pos_dict[tuple[1]].append(tuple[0])
    return pos_dict

def augment(args, word_list, pos_tag_tuple_list, pos_dict):
    new_word_list = [word for word in word_list]
    for i in range(len(word_list)):
        #step1
        x1 = np.random.uniform(0, 1)
        if x1 < args.p_mask:
            new_word_list[i] = "[MASK]"
        elif x1 >= args.p_mask and x1 < (args.p_mask + args.p_pos):
            new_word_list[i] = random.choice(pos_dict[pos_tag_tuple_list[i][1]])
    # step2
    x2 = np.random.uniform(0, 1)
    n_list = [1, 2, 3, 4, 5]
    if x2 < args.p_ng:
        n = random.choice(n_list)
        if n >= len(word_list):
            pass
        else:
            # n-gram
            start_position = random.choice(np.arange(0, len(word_list) - n + 1))
            new_word_list = new_word_list[start_position:start_position + n]
    sentence_after_augment = " ".join(new_word_list)
    return sentence_after_augment

def main(args):
    sample_list = get_samples_from_text(args.train_data_path)
    pos_dict = get_poc_tag_dict(args.train_data_path)
    output_list = []
    for sentence, label in tqdm(sample_list):
        sent_augment_list = []
        word_list = word_tokenize(sentence)
        pos_tag_tuple_list = nltk.pos_tag(word_list)
        for iter in range(args.n_iter):
            new_sentence = augment(args, word_list, pos_tag_tuple_list, pos_dict)
            if new_sentence not in sent_augment_list:
                sent_augment_list.append(new_sentence)
            else:
                continue
        output_list.extend(sent_augment_list)
    random.shuffle(output_list)
    with open(args.output_path, "w") as fout:
        for s in output_list:
            fout.write(str(s) + "\t" + str(random.choice([0, 1])) + "\n")
    pass


if __name__ == "__main__":
    _args = args()
    main(_args)