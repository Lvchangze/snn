import sys
sys.path.append("..")

import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from dataset import RateDataset
from snntorch import spikegen


class RateEncoder():
    def __init__(self, tensor_dataset_path) -> None:
        super(RateEncoder, self).__init__()
        self.tensor_dataset_path = tensor_dataset_path

    def encode(self, num_steps=20):
        rate_data = []
        with open(self.tensor_dataset_path, "rb") as f:
            dataset = pickle.load(f)
        for i in tqdm(range(int(len(dataset)/1))):
            tuple = dataset[i]
            # old_embedding = torch.reshape(tuple[0], (-1, 3000))
            # tmp = (torch.tensor(spikegen.rate(old_embedding, num_steps=num_steps), dtype=torch.float), tuple[1])
            tmp = (torch.tensor(spikegen.rate(tuple[0], num_steps=num_steps), dtype=torch.float), tuple[1])
            rate_data.append(tmp)
        
        rate_dataset = RateDataset(rate_data)

        file_name = f'data/{self.tensor_dataset_path.split(".")[0]}.rate_dataset'
        if not os.path.exists(file_name):
            with open(file_name, 'wb') as f:
                pickle.dump(rate_dataset, f, -1)



if __name__ == "__main__":
    rate_encoder = RateEncoder(tensor_dataset_path="../data/sst2/train_u_3v_sst2_glove100d.tensor_dataset")
    rate_encoder.encode(num_steps=20)
