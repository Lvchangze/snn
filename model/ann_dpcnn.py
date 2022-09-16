import torch
import torch.nn as nn
import numpy as np

class ANN_DPCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv_region = nn.Conv2d(1, args.filter_num, (3, args.hidden_dim), stride=1, bias=False)
        self.conv = nn.Conv2d(args.filter_num, args.filter_num, (3, 1), stride=1)
        # self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(args.filter_num, args.label_num)
        # 定义dropout, 参数为dropout的概率，默认为0.5
        self.dropout = nn.Dropout(args.dropout_p)
        self.bn_1 = nn.BatchNorm2d(num_features=args.filter_num)
        self.bn_2 = nn.BatchNorm2d(num_features=args.filter_num)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze(dim=2)  # [batch_size, num_filters(250)]
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def _block(self, x):
        x = self.padding2(x)
        # px = self.max_pool(x)
        px = self.avg_pool(x)

        x = self.padding1(px)
        x = nn.functional.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = nn.functional.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x