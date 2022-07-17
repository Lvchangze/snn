import numpy as np
import argparse
import pickle
import torch
import snntorch as snn
import torch.nn as nn
from snntorch import backprop
import snntorch.functional as SF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import RateDataset
from snntorch import spikegen
from snntorch import surrogate

label_num = 2
epochs = 10
batch_size = 32
sentence_length = 30
hidden_dim = 100
num_steps = 10
filters = [3,4,5]
filter_num = 100
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self, spike_grad=surrogate.fast_sigmoid(slope=25)) -> None:
        super().__init__()
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=filter_num, kernel_size=(filter_size, hidden_dim))
            for filter_size in filters
        ])
        self.maxpool_1 = nn.ModuleList([
            nn.MaxPool2d((sentence_length - filter_size + 1, 1)) for filter_size in filters
        ])
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc_1 = nn.Linear(len(filters)*filter_num, label_num)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=1)
        conv_out = [conv(x) for conv in self.convs_1]
        pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
        spks = [self.lif1(pooled) for pooled in pooled_out]
        # spks = map(itemgetter(0), spk_mem)
        # mems = map(itemgetter(1), spk_mem)
        spks_1 = torch.cat(spks, dim=1).view(batch_size, -1)
        cur2 = self.fc_1(spks_1)
        spk2, mem2 = self.lif2(cur2)
        return spk2, mem2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# num_inputs = 30 * 100
# num_hidden = 4000
# num_outputs = 2

# # Temporal Dynamics
# num_steps = 20
# beta = 0.95
# spike_grad = surrogate.fast_sigmoid(slope=25)
# class Net(nn.Module):
    # def __init__(self):
    #     super().__init__()

    #     # Initialize layers
    #     self.fc1 = nn.Linear(num_inputs, num_hidden)
    #     self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    #     self.fc2 = nn.Linear(num_hidden, num_outputs)
    #     self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    # # for one time step
    # def forward(self, x):
    #     # Initialize hidden states at t=0
    #     mem1 = self.lif1.init_leaky()
    #     mem2 = self.lif2.init_leaky()

    #     cur1 = self.fc1(x)
    #     spk1, mem1 = self.lif1(cur1, mem1)
    #     cur2 = self.fc2(spk1)
    #     spk2, mem2 = self.lif2(cur2, mem2)
    #     # spk2[:,-1] is the logit in this step
    #     return spk2[:,-1], mem2


if __name__ == "__main__":
    rate_data = []
    with open("data/u_3v_sst_2_glove100d.tensor_dataset", "rb") as f:
        dataset = pickle.load(f)
    for i in tqdm(range(int(len(dataset)/1))):
        tuple = dataset[i]
        old_embedding = torch.reshape(tuple[0], (-1, 3000))
        # tmp = (torch.tensor(spikegen.rate(old_embedding, num_steps=num_steps), dtype=torch.float), tuple[1])
        tmp = (torch.tensor(spikegen.rate(tuple[0], num_steps=num_steps), dtype=torch.float), tuple[1])
        rate_data.append(tmp)
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4 * 2, betas=(0.9, 0.999))
    loss_fn = SF.ce_rate_loss()
    num_epochs = 10
    batch_size = 64

    rate_dataset = RateDataset(rate_data)
    train_loader = DataLoader(rate_dataset, shuffle=True, batch_size=batch_size)

    for epoch in tqdm(range(num_epochs)):
        avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn,
                            num_steps=False, time_var=True, time_first=False, device=device)
        print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")

    # save
    with open("textcnn.model", 'wb') as f:
        pickle.dump(net, f, -1)