import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm
from dataset import RateDataset
from data_preprocess.green_encoder import GreenEncoder
from utils.public import set_seed, save_model_to_file, output_message
from args import SNNArgs
import pickle
from snntorch.utils import reset
from snntorch.backprop import BPTT
import snntorch.functional as SF
from snntorch import spikegen
import snntorch.surrogate as surrogate
from model import TextCNN
import numpy as np
from utils.filecreater import FileCreater
from utils.monitor import Monitor

def build_environment(args: SNNArgs):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    return

def build_dataset(args: SNNArgs, split='train'):
    output_message("Build dataset...")
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
    output_message("Build rated_dataset...")
    if split == 'train':
        assert hasattr(args, "train_dataset")
        dataset = args.train_dataset
    elif split == 'test':
        assert hasattr(args, 'test_dataset')
        dataset = args.test_dataset
    rated_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        if args.encode == "rate":
            tmp = (spikegen.rate(item[0], num_steps=args.num_steps).clone().detach().float(), item[1])
        elif args.encode == "latency":
            tmp = (spikegen.latency(item[0], num_steps=args.num_steps).clone().detach().float(), item[1])
        rated_data.append(tmp)
    rated_dataset = RateDataset(rated_data)
    setattr(args, f'{split}_rated_dataset', rated_dataset)
    # args.rated_dataset = rated_dataset
    return

def build_codebooked_dataset(args: SNNArgs, split='train'):
    output_message("Build codebooked dataset...")
    if split == 'train':
        assert hasattr(args, "train_dataset")
        dataset = args.train_dataset
    elif split == 'test':
        assert hasattr(args, 'test_dataset')
        dataset = args.test_dataset
    codebooked_data = []
    encoder = GreenEncoder(args)
    encoder.write_codebook()
    for i in range(len(dataset)):
        item = dataset[i]
        tmp = (encoder.spike_gen(item[0]).clone().detach(), item[1])
        codebooked_data.append(tmp)
    codebooked_dataset = RateDataset(codebooked_data)
    setattr(args, f'{split}_codebooked_dataset', codebooked_dataset)
    return

def build_dataloader(args: SNNArgs, dataset, split='train'):
    output_message("Build dataloader...")
    # if not hasattr(args, f'{split}_rated_dataset') or not hasattr(args, f'{split}_dataset'):
    #     raise Exception("No such dataset!")
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
    elif args.surrogate == 'atan':
        args.spike_grad = surrogate.atan()
    elif args.surrogate == 'spike_rate_escape':
        args.spike_grad = surrogate.spike_rate_escape()
    elif args.surrogate == 'straight_through_estimator':
        args.spike_grad = surrogate.straight_through_estimator()
    elif args.surrogate == 'triangular':
        args.spike_grad = surrogate.triangular()
    return

def build_criterion(args: SNNArgs):
    if args.ensemble is False:
        if args.loss == 'ce_rate':
            args.loss_fn = SF.ce_rate_loss()
        elif args.loss == 'ce_temporal':
            args.loss_fn = SF.ce_temporal_loss()
        elif args.loss == 'ce_count':
            args.loss_fn = SF.ce_count_loss()
        elif args.loss == 'mse_count':
            args.loss_fn = SF.mse_count_loss()
        elif args.loss == 'mse_temporal':
            args.loss_fn = SF.mse_temporal_loss()
    else:
        if args.loss == 'ce_count':
            args.loss_fn = SF.ce_count_loss(population_code=True, num_classes=2)
        elif args.loss == "mse_count":
            args.loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=2)
    return

def build_model(args: SNNArgs):
    output_message("Build model...")
    args.model = TextCNN(args, spike_grad=args.spike_grad).to(args.device)
    args.model.initial()
    return

def build_optimizer(args: SNNArgs):
    output_message("Build Optimizer...")
    if args.optimizer_name == "Adamw":
        args.optimizer = AdamW(args.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optimizer_name == "SGD":
        args.optimizer = SGD(args.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    return

def bulid_scheduler(args: SNNArgs):
    output_message("Build scheduler...")
    args.scheduler = StepLR(args.optimizer, step_size = 2 , gamma = 0.1)
    return

def predict_accuracy(args, dataloader, model, num_steps, population_code=False, num_classes=False):
    def forward_pass(net, num_steps, data):
        mem_rec = []
        spk_rec = []
        reset(net)  # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            _, spk_out, mem_out = net(data.transpose(1, 0)[step])
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

            if population_code:
                acc += SF.accuracy_rate(spk_rec, targets, population_code=True, num_classes=num_classes) * spk_rec.size(1)
            else:
                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total


def main(args):
    build_dataset(args=args)
    build_dataset(args=args, split='test')
    if args.use_codebook == 'False':
        build_rated_dataset(args)
        build_dataloader(args=args, dataset=args.train_rated_dataset)
        build_rated_dataset(args, split='test')
        build_dataloader(args=args, dataset=args.test_rated_dataset, split='test')
    else:
        build_codebooked_dataset(args=args)
        build_dataloader(args=args, dataset=args.train_codebooked_dataset)
        build_codebooked_dataset(args, split='test')
        build_dataloader(args=args, dataset=args.test_codebooked_dataset, split='test')    

    build_surrogate(args=args)
    build_model(args)
    build_optimizer(args)
    build_criterion(args)
    bulid_scheduler(args)
    dead_neuron_rate_list = []
    output_message("Training Begin")
    too_activate_neuron_rate_list = []
    acc_list = []
    for epoch in tqdm(range(args.epochs)):
        if args.dead_neuron_checker == "True":
            Monitor._EPOCH = epoch
            Monitor.create_epoch_monitor()
        dead_neuron_rate, too_activate_neuron_rate, avg_loss = BPTT(args.model, args.train_dataloader, optimizer=args.optimizer, criterion=args.loss_fn, 
                        num_steps=False, time_var=True, time_first=False, device=args.device)
        dead_neuron_rate_list.append(dead_neuron_rate)
        too_activate_neuron_rate_list.append(too_activate_neuron_rate)
        output_message("Dead_neuron_rate in epoch {}: {}.".format(epoch, dead_neuron_rate))
        output_message("Too_Activate_neuron_rate in epoch {}: {}.".format(epoch, too_activate_neuron_rate))
        output_message("Training epoch {}, avg_loss: {}.".format(epoch, avg_loss))
        # saved_path = FileCreater.build_saving_file(args,description="-epoch{}".format(epoch))
        # save_model_to_file(save_path=saved_path, model=args.model)
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=2)
        output_message("Test acc in epoch {} is: {}".format(epoch, acc))
        acc_list.append(acc)
        if args.dead_neuron_checker == "True":
            Monitor.print_results_by_epoch(epoch)
    output_message("Mean Dead_neuron_rate: {}".format(np.mean(dead_neuron_rate_list)))
    output_message("Mean Too_Activate_neuron_rate: {}".format(np.mean(too_activate_neuron_rate_list)))
    output_message("Best Acc: {}".format(np.max(acc_list)))
    return

if __name__ == "__main__":
    args = SNNArgs.parse()
    build_environment(args)
    FileCreater.build_directory(args, args.logging_dir, 'logging', args.args_for_logging)
    FileCreater.build_directory(args, args.saving_dir, 'saving', args.args_for_logging)
    FileCreater.build_logging(args)
    output_message("Program args: {}".format(args))
    main(args)
