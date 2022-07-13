import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data: str):
        super(MyDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sentence = self.data[index][0]
        label = int(self.data[index][1])
        return sentence, label
