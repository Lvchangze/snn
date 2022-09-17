import torch
import torch.nn as nn
import numpy as np
import snntorch.surrogate as surrogate
import snntorch as snn

class SNN_DPCNN(nn.Module):
    def __init__(self, args, spike_grad=surrogate.fast_sigmoid(slope=25)):
        super().__init__()
        self.label_num = args.label_num
        self.mode = args.mode
        self.conv_region = nn.Conv2d(1, args.filter_num, (args.dpcnn_step_length, args.hidden_dim), stride=1, bias=False)
        self.conv_1 = nn.Conv2d(args.filter_num, args.filter_num, (args.dpcnn_step_length, 1), stride=1, bias=False)
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0)
        self.conv_2 = nn.Conv2d(args.filter_num, args.filter_num, (args.dpcnn_step_length, 1), stride=1, bias=False)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0)
        
        self.conv_list = nn.ModuleList([
            nn.Conv2d(args.filter_num, args.filter_num, (args.dpcnn_step_length, 1), stride=1, bias=False)
            for _ in range(args.dpcnn_block_num)
        ])

        self.lif_list = nn.ModuleList([
            snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0)
            for _ in range(args.dpcnn_block_num)
        ])
        self.avg_pool = nn.AvgPool2d(kernel_size=(args.dpcnn_step_length, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 2, 2))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 2))  # bottom
        self.fc = nn.Linear(args.filter_num, args.label_num, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=args.filter_num)
        self.bn_2 = nn.BatchNorm2d(num_features=args.filter_num)
        self.lif_output = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0, output=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.float()
        x = x.unsqueeze(1)

        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.bn_1(x)
        x = self.lif1(x)

        x = self.conv_1(x)
        x = self.padding1(x)
        x = self.bn_2(x)
        spks = self.lif2(x)

        x = self.conv_2(spks)

        while x.size()[2] >= 2:
            x = self._block(x)
        
        x = x.view(batch_size, -1)
        x = self.fc(x)
        spk2, mem2 = self.lif_output(x)
        
        return spks, spk2, mem2

    
    def _block(self, x):
        x = self.padding2(x)              # padding前：32 * 768 * 21 * 1，padding后：32 * 768 * 23 * 1
        # px = self.max_pool(x)
        px = self.avg_pool(x)             # pool后：32 * 768 * 10 * 1

        for i in range(len(self.conv_list)):
            conv = self.conv_list[i]
            lif = self.lif_list[i]
            x = self.padding1(px)
            x = lif(x)                    # 第一轮，32 * 768 * 14 * 1
            x = conv(x)                   # x第一轮变为 32 * 768 * 10 * 1
        # # Short Cut
        # x = x + px
        return x