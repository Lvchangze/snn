import torch
import torch.nn as nn
import numpy as np

class ANN_TextCNN(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=args.filter_num, kernel_size=(filter_size, args.hidden_dim), bias=False)
            for filter_size in args.filters
        ])
        self.middle_relu = nn.ModuleList([
            nn.ReLU()
            for _ in args.filters
        ])
        self.avgpool_1 = nn.ModuleList([
            nn.AvgPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        ])
        # self.maxpool_1 = nn.ModuleList([
        #     nn.MaxPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        # ])
        self.relu_2 = nn.ReLU()
        self.drop = nn.Dropout(p=args.dropout_p)
        self.fc_1 = nn.Linear(len(args.filters)*args.filter_num, args.label_num, bias=False)
        
    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=1)
        conv_out = [conv(x) for conv in self.convs_1]
        conv_out = [self.middle_relu[i](conv_out[i]) for i in range(len(self.middle_relu))]
        # conv_out = [self.relu_1(i) for i in conv_out]
        # pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
        pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
        pooled_out = [self.relu_2(pool) for pool in pooled_out]
        flatten = torch.cat(pooled_out, dim=1).view(batch_size, -1)
        flatten = self.drop(flatten)
        fc_output = self.fc_1(flatten)
        return fc_output