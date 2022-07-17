import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from dataset import RateDataset
from utils.public import set_seed
from args import SNNArgs
import pickle
from snntorch.backprop import BPTT
import snntorch.functional as SF
from snntorch import spikegen
import snntorch.surrogate as surrogate
from model import TextCNN

def build_environment(args: SNNArgs):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    return

def build_dataset(args: SNNArgs, split='train'):
    if not os.path.exists(args.data_path):
        raise Exception("dataset file not exist!")
    with open(args.data_path, 'rb') as f:
        args.dataset = pickle.load(f)
    return

def build_rated_dataset(args: SNNArgs, split='train'):
    assert hasattr(args, "dataset")
    rated_data = []
    for i in range(128):
        item = args.dataset[i]
        tmp = (torch.tensor(spikegen.rate(item[0], num_steps=args.num_steps), dtype=torch.float), item[1])
        rated_data.append(tmp)
    rated_dataset = RateDataset(rated_data)
    args.rated_dataset = rated_dataset
    return


def build_dataloader(args: SNNArgs, dataset, split='train'):
    if split == 'train':
        args.train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
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

def main(args):
    build_dataset(args=args)
    build_rated_dataset(args)
    build_dataloader(args=args, dataset=args.rated_dataset)
    build_surrogate(args=args)
    build_model(args)
    build_optimizer(args)
    build_criterion(args)
    
    for epoch in tqdm(range(args.epochs)):
        avg_loss = BPTT(args.model, args.train_dataloader, optimizer=args.optimizer, criterion=args.loss_fn,
                            num_steps=False, time_var=True, time_first=False, device=args.device)

    return

if __name__ == "__main__":
    args = SNNArgs.parse(True)
    build_environment(args)
    main(args)
