import torch
import torch.nn as nn
import numpy as np
import snntorch.surrogate as surrogate
import snntorch as snn

class SNN_BiLSTM(nn.Module):
    def __init__(self, args, spike_grad=surrogate.fast_sigmoid(slope=25)) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(batch_first=True, input_size=args.hidden_dim, hidden_size=args.lstm_hidden_size, num_layers=args.lstm_layers_num, bidirectional=bool(args.bidirectional))
        
        if args.bidirectional == "True":
            self.fc_1 = nn.Linear(args.lstm_hidden_size * 2, args.lstm_fc1_num)
        else:
            self.fc_1 = nn.Linear(args.lstm_hidden_size, args.lstm_fc1_nume)
        
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0)
        self.output_fc = nn.Linear(args.lstm_fc1_num, args.label_num)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0, output=True)
        
    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        output, (hidden,cell) = self.lstm(x)
        x = self.fc_1(output)
        spks = self.lif1(x)
        fc_output = self.output_fc(spks)            # batch_size * seq_len * embed_dim
        fc_output = fc_output[:,-1,:].squeeze(1)   # batch_size * label_num
        spk2, mem2 = self.lif2(fc_output)
        return spks, spk2, mem2