from email import header
import random
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

import csv

def main(file_path, output_path):
    with open(file_path, "r") as fin, open(output_path, "w") as fout:
        f_csv = csv.reader(fin)
        headers = next(f_csv)
        output_samples = []
        for row in f_csv:
            output_samples.append("\t".join(row) + "\n")
        random.shuffle(output_samples)
        for sample in output_samples:
            fout.write(sample)
    pass

if __name__ == "__main__":
    file_path = "../data/sst5/val.csv"
    output_path = "../data/sst5/val.txt"
    main(file_path, output_path)