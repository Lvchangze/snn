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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_inputs = 30 * 100
num_hidden = 4000
num_outputs = 2

# Temporal Dynamics
num_steps = 20
beta = 0.95
spike_grad = surrogate.fast_sigmoid(slope=25)

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


if __name__ == "__main__":
    rate_data = []
    with open("data/u_3v_sst_2_glove100d.tensor_dataset", "rb") as f:
        dataset = pickle.load(f)
    for i in tqdm(range(int(len(dataset)/100))):
        tuple = dataset[i]
        old_embedding = torch.reshape(tuple[0], (-1, 3000))
        tmp = (torch.tensor(spikegen.rate(old_embedding, num_steps=num_steps), dtype=torch.float), tuple[1])
        rate_data.append(tmp)

    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
    loss_fn = SF.ce_rate_loss()
    num_epochs = 10
    batch_size = 128

    rate_dataset = RateDataset(rate_data)
    train_loader = DataLoader(rate_dataset, shuffle=True, batch_size=batch_size)


    for epoch in tqdm(range(num_epochs)):
        avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn,
                            num_steps=False, time_var=True, time_first=False, device=device)
        print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")