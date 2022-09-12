from email.policy import strict
import os
from random import choices
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch.nn.functional as F
from tqdm import tqdm
from dataset import RateDataset
from data_preprocess.green_encoder import GreenEncoder
from data_preprocess.embedding_encoder import EmbeddingEncoder
from utils.public import set_seed, save_model_to_file, output_message, load_model_from_file, clean_tokenize
from args import SNNArgs
import pickle
from snntorch.utils import reset
from snntorch.backprop import BPTT
import snntorch.functional as SF
from snntorch import spikegen
import snntorch.surrogate as surrogate
from model import TextCNN, ANN_TextCNN
import numpy as np
from utils.filecreater import FileCreater
from utils.monitor import Monitor
from textattack import Attacker
from utils.attackutils import CustomTextAttackDataset, build_attacker
from textattack.models.wrappers.snn_model_wrapper import SNNModelWrapper
from textattack.models.wrappers.ann_model_wrapper import ANNModelWrapper
from textattack.loggers import AttackLogManager, attack_log_manager
from utils.metrics import SimplifidResult
import textattack
import time
from dataset import TensorDataset

def build_environment(args: SNNArgs):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_seed == "True":
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
        tmp = (encoder.spike_gen(item[0], num_step=args.num_steps).clone().detach(), item[1])
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
    if args.model_mode == "ann":
        args.loss_fn = F.cross_entropy
    else:
        if args.ensemble == 'False':
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
            elif args.loss == "ce_rate":
                args.loss_fn = SF.ce_rate_loss(population_code=True, num_classes=2)
            elif args.loss == "mse_count":
                args.loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=2)
        
    return

def build_model(args: SNNArgs):
    output_message("Build model...")
    if args.model_mode == "ann":
        args.model = ANN_TextCNN(args).to(args.device)
    elif args.model_mode == "snn":
        args.model = TextCNN(args, spike_grad=args.spike_grad).to(args.device)
        args.model.initial()
    elif args.mode == "conversion":
        args.model = TextCNN(args, spike_grad=args.spike_grad).to(args.device)
        args.model.load_state_dict(torch.load(args.conversion_model_path), strict=False)
        args.model.initial()
    return

def build_optimizer(args: SNNArgs):
    output_message("Build Optimizer...")
    if args.optimizer_name == "Adamw":
        args.optimizer = AdamW(args.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optimizer_name == "SGD":
        args.optimizer = SGD(args.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
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

def train(args):
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

    if args.mode == "conversion":
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=2)
        output_message("Test acc of initial conversion TextCNN is: {}".format(acc))

    build_optimizer(args)
    build_criterion(args)
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
        saved_path = FileCreater.build_saving_file(args,description="-epoch{}".format(epoch))
        save_model_to_file(save_path=saved_path, model=args.model)
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=2)
        output_message("Test acc in epoch {} is: {}".format(epoch, acc))
        acc_list.append(acc)
        if args.dead_neuron_checker == "True":
            Monitor.print_results_by_epoch(epoch)
    output_message("Mean Dead_neuron_rate: {}".format(np.mean(dead_neuron_rate_list)))
    output_message("Mean Too_Activate_neuron_rate: {}".format(np.mean(too_activate_neuron_rate_list)))
    output_message("Best Acc: {}".format(np.max(acc_list)))
    return

def attack(args: SNNArgs):
    if args.model_mode == 'snn':
        build_surrogate(args=args)
        build_model(args)
        args.tokenizer = EmbeddingEncoder(args.vocab_path, args.data_dir, args.max_len, model_mode="snn")
        model_wrapper = SNNModelWrapper(args, args.model, args.tokenizer)
    elif args.model_mode == 'ann':
        build_model(args)
        args.tokenizer = EmbeddingEncoder(args.vocab_path, args.data_dir, args.max_len, model_mode="ann")
        model_wrapper = ANNModelWrapper(args, args.model, args.tokenizer)
    attack = build_attacker(args, model_wrapper)
    # test_instances = EmbeddingEncoder.dataset_encode('data/sst2/test.txt')
    load_model_from_file(args.attack_model_path, args.model)
    attack_log_dir = FileCreater.build_directory(args, args.attack_logging_dir, 'attacking', args.args_for_logging)
    attack_log_path = os.path.join(attack_log_dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    attack_log_manager = AttackLogManager()
    attack_log_manager.add_output_file(attack_log_path, "ansi")
    # test_instances = [x for x in test_instances if len(x[0].split(' ')) > 4]
    for i in range(args.attack_times):
        print("Attack time {}".format(i))
        # total_len = len(test_instances)
        # attack_num = min(total_len, args.attack_numbers)
        # choices_arr = np.arange(total_len)
        # np.random.shuffle(choices_arr)
        # choice_instances = []
        # for item in choices_arr[:attack_num]:
        #     choice_instances.append(test_instances[item])
        # dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances)
        dataset = textattack.datasets.HuggingFaceDataset("sst2", split="validation", shuffle=True)
        attack_num = min(args.attack_numbers, len(dataset))
        # attack_args = textattack.AttackArgs(num_examples=attack_num,log_to_csv=attack_log_dir+"/{}_log.csv".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
        #     checkpoint_interval=100,checkpoint_dir="checkpoints",disable_stdout=True)
        attack_args = textattack.AttackArgs(num_examples=attack_num,log_to_csv=attack_log_dir+"/{}_log.csv".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            disable_stdout=True)
        attacker = Attacker(attack, dataset, attack_args)
        results_iterable = attacker.attack_dataset()
        description = tqdm(results_iterable, total=attack_num)
        result_statistics = SimplifidResult()
        for result in description:
            # try:
            attack_log_manager.log_result(result)
            result_statistics(result)
            description.set_description(result_statistics.__str__())
            # except Exception as e:
            #     print(e)
            #     print('error in process')
            #     continue
    attack_log_manager.enable_stdout()
    attack_log_manager.log_summary()
    pass
    
def ann_train(args):
    def get_tensor_dataset(file):
        glove_dict = {}
        with open(args.vocab_path, "r") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        zero_embedding = np.array([0] * args.hidden_dim, dtype=float)

        sample_list = []
        with open(file, "r") as f:
            for line in f.readlines():
                temp = line.split('\t')
                sentence = temp[0].strip()
                label = int(temp[1])
                sample_list.append((sentence, label))

        embedding_tuple_list = []
        for i in range(len(sample_list)):
            sent_embedding = np.array([[0] * args.hidden_dim] * args.sentence_length, dtype=float)
            # text_list = sample_list[i][0].split()
            text_list = clean_tokenize(sample_list[i][0])
            label = sample_list[i][1]
            for j in range(args.sentence_length):
                if j >= len(text_list):
                    embedding = zero_embedding # zero padding
                else:
                    word = text_list[j]
                    embedding = glove_dict[word] if word in glove_dict.keys() else zero_embedding
                sent_embedding[j] = embedding
            embedding_tuple_list.append((torch.tensor(sent_embedding), label))
        dataset = TensorDataset(embedding_tuple_list)
        return dataset
    
    # build_dataset(args=args)
    # build_dataset(args=args, split='test')
    # build_dataloader(args=args, dataset=args.train_dataset)
    # build_dataloader(args=args, dataset=args.test_dataset, split='test')

    build_model(args)
    build_optimizer(args)
    build_criterion(args)

    build_dataloader(args=args, dataset=get_tensor_dataset("data/sst2/train.txt"))
    test_dataset = get_tensor_dataset("data/sst2/dev.txt")
    build_dataloader(args=args, dataset=test_dataset, split='test')

    acc_list = []
    for epoch in tqdm(range(args.epochs)):
        for data, target in args.train_dataloader:
            args.model.train()
            data = data.to(args.device)
            target = target.to(args.device)
            output = args.model(data)
            loss = args.loss_fn(output, target)
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()
        saved_path = FileCreater.build_saving_file(args, description="-epoch{}".format(epoch))
        save_model_to_file(save_path=saved_path, model=args.model)
        args.model.eval()
        with torch.no_grad():
            correct = 0
            for data, y_batch in args.test_dataloader:
                data = data.to(args.device)
                y_batch = y_batch.to(args.device)
                output = args.model(data)
                correct += int(y_batch.eq(torch.max(output,1)[1]).sum())
        acc_list.append(float(correct/len(test_dataset)))
        output_message(f"Epoch {epoch} Acc: {float(correct/len(test_dataset))}")
    output_message(np.max(acc_list))
    pass

if __name__ == "__main__":
    args = SNNArgs.parse()
    build_environment(args)
    FileCreater.build_directory(args, args.logging_dir, 'logging', args.args_for_logging)
    FileCreater.build_directory(args, args.saving_dir, 'saving', args.args_for_logging)
    FileCreater.build_logging(args)
    output_message("Program args: {}".format(args))
    if args.mode == 'train' and args.model_mode == "snn":
        train(args)
    elif args.mode == 'attack':
        attack(args)
    elif args.mode == 'train' and args.model_mode == "ann":
        ann_train(args)
    elif args.mode == "conversion":
        train(args)
