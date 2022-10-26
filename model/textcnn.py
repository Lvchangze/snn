import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch.surrogate as surrogate
import snntorch as snn
import numpy as np
from utils.config import INITIAL_MEAN_DICT
from utils.monitor import Monitor

class SNN_TextCNN(nn.Module):
    def __init__(self, args, spike_grad=surrogate.fast_sigmoid(slope=25)) -> None:
        super().__init__()
        self.dead_neuron_checker = args.dead_neuron_checker
        self.initial_method = args.initial_method
        self.positive_init_rate = args.positive_init_rate
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=args.filter_num, kernel_size=(filter_size, args.hidden_dim))
            for filter_size in args.filters
        ])
        self.middle_lifs = nn.ModuleList([
            snn.Leaky(beta=args.beta, spike_grad = spike_grad, init_hidden=True, threshold=args.threshold)
            for _ in args.filters
        ])
        self.avgpool_1 = nn.ModuleList([
            nn.AvgPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        ])
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=args.threshold)
        self.fc_1 = nn.Linear(len(args.filters)*args.filter_num, args.label_num)
        # self.fc_2 = nn.Linear(args.hidden_layer_num, args.label_num)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=args.threshold, output=True)
    
    def initial(self):
        for c in self.convs_1:
            c.weight.data.add_(INITIAL_MEAN_DICT['conv-kaiming'][self.positive_init_rate])
        m = self.fc_1
        m.weight.data.add_(INITIAL_MEAN_DICT["linear-kaiming"][self.positive_init_rate])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=1)
        conv_out = [conv(x) for conv in self.convs_1]

        # conv_out = [self.relu_1(c) for c in conv_out]
        conv_out = [self.middle_lifs[i](conv_out[i]) for i in range(len(self.middle_lifs))]

        pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
        # pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
        
        spks = [self.lif1(pooled) for pooled in pooled_out]
        spks_1 = torch.cat(spks, dim=1).view(batch_size, -1)
        hidden_1 = self.fc_1(spks_1)
        # cur2 = self.fc_2(hidden_1)
        spk2, mem2 = self.lif2(hidden_1)
        if self.dead_neuron_checker == "True":
            temp_spks = spks_1.sum(dim=0)
            Monitor.add_monitor(temp_spks, 0)
        return spks_1, spk2, mem2
        
