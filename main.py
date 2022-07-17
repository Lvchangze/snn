# from ctypes.wintypes import tagRECT
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from dataset import RateDataset
from utils.public import set_seed, save_model_to_file, output_message
from args import SNNArgs
import pickle
from snntorch.utils import reset
from snntorch.backprop import BPTT
import snntorch.functional as SF
from snntorch import spikegen
import snntorch.surrogate as surrogate
from model import TextCNN
import logging
from utils.filecreater import FileCreater

def build_environment(args: SNNArgs):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    return

def build_dataset(args: SNNArgs, split='train'):
    if not os.path.exists(args.data_path):
        raise Exception("dataset file not exist!")
    if split == 'train':
        with open(args.data_path, 'rb') as f:
            args.train_dataset = pickle.load(f)
    elif split == 'test':
        with open(args.test_data_path, 'rb') as f:
            args.test_dataset = pickle.load(f)
    return

def build_rated_dataset(args: SNNArgs, split='train'):
    if split == 'train':
        assert hasattr(args, "train_dataset")
        dataset = args.train_dataset
    elif split == 'test':
        assert hasattr(args, 'test_dataset')
        dataset = args.test_dataset
    rated_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        tmp = (torch.tensor(spikegen.rate(item[0], num_steps=args.num_steps), dtype=torch.float), item[1])
        rated_data.append(tmp)
    rated_dataset = RateDataset(rated_data)
    setattr(args, f'{split}_rated_dataset', rated_dataset)
    # args.rated_dataset = rated_dataset
    return


def build_dataloader(args: SNNArgs, dataset, split='train'):
    if not hasattr(args, f'{split}_rated_dataset') or not hasattr(args, f'{split}_dataset'):
        raise Exception("No such dataset!")
    setattr(args, f'{split}_dataloader', DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    # if split == 'train':
    #     args.train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # elif split == 'test':
    #     args.test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return

def build_surrogate(args: SNNArgs):
    if args.surrogate == 'fast_sigmoid':
        args.spike_grad = surrogate.fast_sigmoid()
    elif args.surrogate == 'sigmoid':
        args.spike_grad = surrogate.sigmoid()
    return

def build_criterion(args: SNNArgs):
    if args.loss == 'cross_entropy':
        args.loss_fn = SF.ce_rate_loss()
    return

def build_model(args: SNNArgs):
    args.model = TextCNN(args, spike_grad=args.spike_grad).to(args.device)
    return

def build_optimizer(args: SNNArgs):
    args.optimizer = AdamW(args.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    return


def predict_accuracy(args, dataloader, model, num_steps):
    def forward_pass(net, num_steps, data):
        mem_rec = []
        spk_rec = []
        reset(net)  # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            spk_out, mem_out = net(data.transpose(1, 0)[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)
    
    with torch.no_grad():
        total = 0
        acc = 0
        model.eval()

        dataloader = iter(dataloader)
        for data, targets in dataloader:
            data = data.to(args.device)
            targets = targets.to(args.device)
            spk_rec, _ = forward_pass(model, num_steps, data)
            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total


def main(args):
    build_dataset(args=args)
    build_rated_dataset(args)
    build_dataloader(args=args, dataset=args.train_rated_dataset)
    build_dataset(args=args, split='test')
    build_rated_dataset(args, split='test')
    build_dataloader(args=args, dataset=args.test_rated_dataset, split='test')
    build_surrogate(args=args)
    build_model(args)
    build_optimizer(args)
    build_criterion(args)
    
    for epoch in tqdm(range(args.epochs)):
        avg_loss = BPTT(args.model, args.train_dataloader, optimizer=args.optimizer, criterion=args.loss_fn,
                            num_steps=False, time_var=True, time_first=False, device=args.device)
        output_message("Training epoch {}, avg_loss: {}.".format(epoch, avg_loss))
        saved_path = FileCreater.build_saving_file(args,description="-epoch{}".format(epoch))
        save_model_to_file(save_path=saved_path, model=args.model)
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps)
        output_message("Test acc in epoch {} is: {}".format(epoch, acc))
    return

if __name__ == "__main__":
    args = SNNArgs.parse()
    build_environment(args)
    FileCreater.build_directory(args, args.logging_dir, 'logging', args.args_for_logging)
    FileCreater.build_directory(args, args.saving_dir, 'saving', args.args_for_logging)
    FileCreater.build_logging(args)
    output_message("Program args: {}".format(args))
    main(args)
