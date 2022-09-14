import torch
import torch.nn as nn
import numpy as np

class ANN_BiLSTM(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=args.hidden_dim, hidden_size=args.lstm_hidden_size, \
            num_layers=args.lstm_layers_num, bidirectional=bool(args.bidirectional), bias=False)
        
        if args.bidirectional:
            self.fc_1 = nn.Linear(args.lstm_hidden_size * 2, args.lstm_fc1_num, bias=False)
        else:
            self.fc_1 = nn.Linear(args.lstm_hidden_size, args.lstm_fc1_num, bias=False)
        
        self.relu = nn.ReLU()
        self.output_fc = nn.Linear(args.lstm_fc1_num, args.label_num, bias=False)
        
    def forward(self, x):
        x = x.float()   
        output, (hidden,cell) = self.lstm(x)
        x = self.fc_1(output)
        x = self.relu(x)
        fc_output = self.output_fc(x)
        fc_output = fc_output[:,-1,:].squeeze(1)
        return fc_output