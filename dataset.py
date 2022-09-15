from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, data: str):
        super(TensorDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        embedding = self.data[index][0]
        label = int(self.data[index][1])
        return embedding, label

class RateDataset(Dataset):
    def __init__(self, data: str):
        super(RateDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        rate_code = self.data[index][0]
        label = int(self.data[index][1])
        return rate_code, label

class TxtDataset(Dataset):
    def __init__(self, data_path: str):
        super(TxtDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        temp = line.split('\t')
        sentence = temp[0]
        label = int(temp[1])
        return sentence, label