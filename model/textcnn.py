import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch.surrogate as surrogate
import snntorch as snn
from operator import itemgetter


class TextCNN(nn.Module):
    def __init__(self, args, spike_grad=surrogate.fast_sigmoid(slope=25)) -> None:
        super().__init__()
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=args.filter_num, kernel_size=(filter_size, args.hidden_dim))
            for filter_size in args.filters
        ])
        self.maxpool_1 = nn.ModuleList([
            nn.MaxPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        ])
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True)
        self.fc_1 = nn.Linear(len(args.filters)*args.filter_num, args.label_num)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, output=True)

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

        
