import torch
import torch.nn as nn
import numpy as np

class ANN_DPCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.label_num = args.label_num
        self.mode = args.mode
        # self.layer_embeddings = []
        self.conv_region = nn.Conv2d(1, args.filter_num, (args.dpcnn_step_length, args.hidden_dim), stride=1, bias=False)
        self.conv_1 = nn.Conv2d(args.filter_num, args.filter_num, (args.dpcnn_step_length, 1), stride=1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(args.filter_num, args.filter_num, (args.dpcnn_step_length, 1), stride=1, bias=False)
        self.relu2 = nn.ReLU()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(args.filter_num, args.filter_num, (args.dpcnn_step_length, 1), stride=1, bias=False)
            for _ in range(args.dpcnn_block_num)
        ])
        self.avg_pool = nn.AvgPool2d(kernel_size=(args.dpcnn_step_length, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 2, 2))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 2))  # bottom
        # self.dropout = nn.Dropout(args.dropout_p)
        self.fc = nn.Linear(args.filter_num, args.label_num, bias=False)
        # self.bn_1 = nn.BatchNorm2d(num_features=args.filter_num)
        # self.bn_2 = nn.BatchNorm2d(num_features=args.filter_num)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.float()
        x = x.unsqueeze(1)

        x = self.conv_region(x)
        # x = self.bn_1(x)
        x = self.padding1(x)
        x = self.relu1(x)

        x = self.conv_1(x)
        # x = self.bn_2(x)
        x = self.padding1(x)
        x = self.relu2(x)

        x = self.conv_2(x)

        while x.size()[2] >= 2:
            x = self._block(x)
        
        fc_input = x.view(batch_size, -1)

        # fc_input = self.dropout(fc_input)

        fc_output = self.fc(fc_input)

        if self.mode == "distill":
            return fc_input, fc_output

        return fc_output
    
    def _block(self, x):
        x = self.padding2(x)
        x = self.avg_pool(x)
        for i in range(len(self.conv_list)):
            conv = self.conv_list[i]
            x = self.padding1(x)
            x = nn.functional.relu(x)
            x = conv(x)
        return x