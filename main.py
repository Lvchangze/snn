from email.policy import strict
import os
from random import choices
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adadelta
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
from model import SNN_TextCNN, ANN_TextCNN, ANN_BiLSTM, SNN_BiLSTM, ANN_DPCNN, SNN_DPCNN, Normal_TextCNN
import numpy as np
from utils.filecreater import FileCreater
from utils.monitor import Monitor
from textattack import Attacker
from utils.attackutils import CustomTextAttackDataset, build_attacker
from textattack.models.wrappers.snn_model_wrapper import SNNModelWrapper
from textattack.models.wrappers.ann_model_wrapper import ANNModelWrapper
from textattack.models.wrappers.snn_population_model_wrapper import SNNPopulationModelWrapper
from textattack.loggers import AttackLogManager, attack_log_manager
from utils.metrics import SimplifidResult
import textattack
import time
from dataset import TensorDataset, TxtDataset
from transformers import BertTokenizer, BertForSequenceClassification

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
    elif split == 'dev':
        with open(args.dev_data_path, 'rb') as f:
            args.dev_dataset = pickle.load(f)
    return

def build_rated_dataset(args: SNNArgs, split='train'):
    output_message("Build rated_dataset...")
    if split == 'train':
        assert hasattr(args, "train_dataset")
        dataset = args.train_dataset
    elif split == 'test':
        assert hasattr(args, 'test_dataset')
        dataset = args.test_dataset
    elif split == 'dev':
        assert hasattr(args, 'dev_dataset')
        dataset = args.dev_dataset
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
    elif split == 'dev':
        assert hasattr(args, 'dev_dataset')
        dataset = args.dev_dataset
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
                args.loss_fn = SF.ce_count_loss(population_code=True, num_classes=args.ensemble_class)
            elif args.loss == "ce_rate":
                args.loss_fn = SF.ce_rate_loss(population_code=True, num_classes=args.ensemble_class)
            elif args.loss == "mse_count":
                args.loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=args.ensemble_class)
        
    return

def build_model(args: SNNArgs):
    output_message("Build model...")
    if args.model_mode == "ann":
        if args.model_type == "textcnn":
            args.model = ANN_TextCNN(args).to(args.device)
        elif args.model_type == "normal_textcnn":
            args.model = Normal_TextCNN(args).to(args.device)
        elif args.model_type == "lstm":
            args.model = ANN_BiLSTM(args).to(args.device)
        elif args.model_type == "dpcnn":
            args.model = ANN_DPCNN(args).to(args.device)
    elif args.model_mode == "snn":
        if args.model_type == "textcnn":
            args.model = SNN_TextCNN(args, spike_grad=args.spike_grad).to(args.device)
            args.model.initial()
        elif args.model_type == "lstm":
            args.model = SNN_BiLSTM(args, spike_grad=args.spike_grad).to(args.device)
        elif args.model_type == "dpcnn":
            args.model = SNN_DPCNN(args, spike_grad=args.spike_grad).to(args.device)
    print(args.model)
    return

def build_optimizer(args: SNNArgs):
    output_message("Build Optimizer...")
    if args.optimizer_name == "Adamw":
        args.optimizer = AdamW(args.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optimizer_name == "SGD":
        args.optimizer = SGD(args.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_name == "Adadelta":
        args.optimizer = Adadelta(args.model.parameters() ,lr = 1.0, rho=0.95, weight_decay=args.weight_decay)
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
    build_surrogate(args=args)
    build_model(args)
    build_dataset(args=args)
    build_dataset(args=args, split='test')
    # build_dataset(args=args, split='dev')
    if args.use_codebook == 'False':
        build_rated_dataset(args)
        build_dataloader(args=args, dataset=args.train_rated_dataset)
        build_rated_dataset(args, split='test')
        build_dataloader(args=args, dataset=args.test_rated_dataset, split='test')
        # build_rated_dataset(args, split='dev')
        # build_dataloader(args=args, dataset=args.dev_rated_dataset, split='dev')
    else:
        build_codebooked_dataset(args=args)
        build_dataloader(args=args, dataset=args.train_codebooked_dataset)
        build_codebooked_dataset(args, split='test')
        build_dataloader(args=args, dataset=args.test_codebooked_dataset, split='test')
        # build_codebooked_dataset(args, split='dev')
        # build_dataloader(args=args, dataset=args.dev_codebooked_dataset, split='dev')    

    if args.mode == "conversion":
        args.model.load_state_dict(torch.load(args.conversion_model_path), strict=False)
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=args.ensemble_class)
        output_message("Test acc of conversioned {} is: {}".format(args.model_type, acc))

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
        
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=args.ensemble_class)
        output_message("Test acc in epoch {} is: {}".format(epoch, acc))
        acc_list.append(acc)

        if acc >= np.max(acc_list):
            saved_path = FileCreater.build_saving_file(args,description="-epoch{}".format(epoch))
            save_model_to_file(save_path=saved_path, model=args.model)
            
        # dev_acc = predict_accuracy(args, args.dev_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=args.ensemble_class)
        # output_message("Dev acc in epoch {} is: {}".format(epoch, dev_acc))
        
        if args.dead_neuron_checker == "True":
            Monitor.print_results_by_epoch(epoch)
    output_message("Mean Dead_neuron_rate: {}".format(np.mean(dead_neuron_rate_list)))
    output_message("Mean Too_Activate_neuron_rate: {}".format(np.mean(too_activate_neuron_rate_list)))
    output_message("Best Test Acc: {}".format(np.max(acc_list)))
    return

def attack(args: SNNArgs):
    if args.model_mode == 'snn':
        build_surrogate(args=args)
        build_model(args)
        args.tokenizer = EmbeddingEncoder(args.vocab_path, args.data_dir, args.max_len, need_norm = True)
        if args.ensemble == 'True':
            model_wrapper = SNNPopulationModelWrapper(args, args.model, args.tokenizer)
        else:
            model_wrapper = SNNModelWrapper(args, args.model, args.tokenizer)
    elif args.model_mode == 'ann':
        build_model(args)
        args.tokenizer = EmbeddingEncoder(args.vocab_path, args.data_dir, args.max_len, need_norm = True)
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
        # dataset = textattack.datasets.HuggingFaceDataset("sst2", split="validation", shuffle=True)
        sample_list = []
        with open(args.attack_text_path, "r") as f:
            for line in f.readlines():
                temp = line.split('\t')
                sentence = temp[0].strip()
                label = int(temp[1])
                sample_list.append((sentence, label))
        dataset = textattack.datasets.Dataset(sample_list)
        attack_num = min(args.attack_numbers, len(dataset))
        # attack_args = textattack.AttackArgs(num_examples=attack_num,log_to_csv=attack_log_dir+"/{}_log.csv".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
        #     checkpoint_interval=100,checkpoint_dir="checkpoints",disable_stdout=True)
        attack_args = textattack.AttackArgs(num_examples=attack_num, log_to_csv=attack_log_dir+"/{}_log.csv".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
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
    
def ann_train(args: SNNArgs):
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
    
    build_dataset(args=args)
    build_dataset(args=args, split='test')
    # build_dataset(args=args, split='dev')
    build_dataloader(args=args, dataset=args.train_dataset)
    build_dataloader(args=args, dataset=args.test_dataset, split='test')
    # build_dataloader(args=args, dataset=args.dev_dataset, split='dev')

    build_model(args)
    build_optimizer(args)
    build_criterion(args)

    # build_dataloader(args=args, dataset=get_tensor_dataset("data/sst2/train.txt"))
    test_dataset = args.test_dataset
    # dev_dataset = args.dev_dataset
    # build_dataloader(args=args, dataset=test_dataset, split='test')

    acc_list = []
    for epoch in tqdm(range(args.epochs)):
        for data, target in args.train_dataloader:
            args.model.train()
            data = data.to(args.device)
            target = target.to(args.device)
            output = args.model(data)
            if args.ensemble == "False":
                loss = args.loss_fn(output, target)
            else:
                tmp = torch.zeros(output.shape[0], args.ensemble_class).to(args.device)
                for idx in range(args.ensemble_class):
                    tmp[:, idx] = output[
                        :,
                        int(args.label_num * idx / args.ensemble_class) : int(
                                args.label_num * (idx + 1) / args.ensemble_class
                        )
                    ].sum(-1)
                # scale loss
                # tmp = tmp / 1
                loss = args.loss_fn(tmp, target)
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()
        
        args.model.eval()
        with torch.no_grad():
            correct = 0
            for data, y_batch in args.test_dataloader:
                data = data.to(args.device)
                y_batch = y_batch.to(args.device)
                output = args.model(data)
                if args.ensemble == "False":
                    correct += int(y_batch.eq(torch.max(output,1)[1]).sum())
                else:
                    tmp = torch.zeros(output.shape[0], args.ensemble_class).to(args.device)
                    for idx in range(args.ensemble_class):
                        tmp[:, idx] = (output[
                            :,
                            int(args.label_num * idx / args.ensemble_class) : int(
                                    args.label_num * (idx + 1) / args.ensemble_class
                            )
                        ].sum(-1)
                        )
                    correct += int(y_batch.eq(torch.max(tmp,1)[1]).sum())
            output_message(f"Epoch {epoch} Acc: {float(correct/len(test_dataset))}")
            acc_list.append(float(correct/len(test_dataset)))
            if float(correct/len(test_dataset)) >= np.max(acc_list):
                saved_path = FileCreater.build_saving_file(args, description="-epoch{}".format(epoch))
                save_model_to_file(save_path=saved_path, model=args.model)
            
            # correct = 0
            # for data, y_batch in args.dev_dataloader:
            #     data = data.to(args.device)
            #     y_batch = y_batch.to(args.device)
            #     output = args.model(data)
            #     if args.ensemble == "False":
            #         correct += int(y_batch.eq(torch.max(output,1)[1]).sum())
            #     else:
            #         tmp = torch.zeros(output.shape[0], args.ensemble_class).to(args.device)
            #         for idx in range(args.ensemble_class):
            #             tmp[:, idx] = (output[
            #                 :,
            #                 int(args.label_num * idx / args.ensemble_class) : int(
            #                         args.label_num * (idx + 1) / args.ensemble_class
            #                 )
            #             ].sum(-1)
            #             )
            #         correct += int(y_batch.eq(torch.max(tmp,1)[1]).sum())
            # output_message(f"Epoch {epoch} Acc: {float(correct/len(dev_dataset))}")
        
    output_message(f"Best Test Acc: {np.max(acc_list)}")

def conversion(args: SNNArgs):
    if args.conversion_mode == "normalize":
        build_surrogate(args)
        if args.model_type == "lstm":
            args.model = SNN_BiLSTM(args, spike_grad=args.spike_grad).to(args.device)
        elif args.model_type == "textcnn":
            args.model = SNN_TextCNN(args, spike_grad=args.spike_grad).to(args.device)
        elif args.model_type == "dpcnn":
            args.model = SNN_DPCNN(args, spike_grad=args.spike_grad).to(args.device)
        
        build_dataset(args=args, split='test')

        if args.use_codebook == 'False':
            build_rated_dataset(args, split='test')
            build_dataloader(args=args, dataset=args.test_rated_dataset, split='test')
        else:
            build_codebooked_dataset(args, split='test')
            build_dataloader(args=args, dataset=args.test_codebooked_dataset, split='test')   

        
        saved_weights = torch.load(args.conversion_model_path)
        args.model.load_state_dict(saved_weights, strict=False)

        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=args.ensemble_class)
        output_message("Test acc of conversioned {} without normalize is: {}".format(args.model_type, acc))

        # if args.conversion_normalize_type == "model_base":
        for key in saved_weights.keys():
            # default: fc is output layer
            if str(key).endswith("weight") and (not str(key).startswith("fc")) and (not str(key).startswith("output")):
                max_input_wt = torch.max(saved_weights[key])
                if max_input_wt > 0.0:
                    saved_weights[key] = saved_weights[key] / max_input_wt
        args.model.load_state_dict(saved_weights, strict=False)
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=args.ensemble_class)
        output_message("Test acc of conversioned {} after model_based normalize is: {}".format(args.model_type, acc))
        
        saved_weights = torch.load(args.conversion_model_path)
        # elif args.conversion_normalize_type == "data_base":
        output_layer_factor = 1.19
        convs_layer_factor = 1.0021
        for key in saved_weights.keys():
            # default: fc is output layer
            if str(key).startswith("fc") or str(key).startswith("output"):
                saved_weights[key] = saved_weights[key] / output_layer_factor
            elif str(key).endswith("weight"):
                saved_weights[key] = saved_weights[key] / convs_layer_factor
        args.model.load_state_dict(saved_weights, strict=False)
        acc = predict_accuracy(args, args.test_dataloader, args.model, args.num_steps, population_code=bool(args.ensemble), num_classes=args.ensemble_class)
        output_message("Test acc of conversioned {} after data_based normalize is: {}".format(args.model_type, acc))
    
    elif args.conversion_mode == "tune":
        train(args)
    pass

def distill(args: SNNArgs):
    teacher_tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model_path, num_labels=args.label_num, output_hidden_states=True).to(args.device)
    for param in teacher_model.parameters():
        param.requires_grad = False

    if args.student_model_name == "lstm":
        student_model = ANN_BiLSTM(args).to(args.device)
    elif args.student_model_name == "textcnn":
        student_model = ANN_TextCNN(args).to(args.device)
    elif args.student_model_name == "dpcnn":
        student_model = ANN_DPCNN(args).to(args.device)
    optimizer = Adadelta(student_model.parameters() ,lr = 1.0, rho=0.95)
    teacher_data_loader = DataLoader(dataset=TxtDataset(data_path=args.data_augment_path), batch_size= args.distill_batch, shuffle=False)
    
    build_dataset(args=args, split='test')
    build_dataloader(args=args, dataset=args.test_dataset, split='test')
    # build_dataset(args=args, split='dev')
    # build_dataloader(args=args, dataset=args.dev_dataset, split='dev')
    test_dataset = args.test_dataset
    # dev_dataset = args.dev_dataset

    def to_device(x, device):
        for key in x:
            x[key] = x[key].to(device)
    def get_dict():
        glove_dict = {}
        with open(args.vocab_path, "r") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        hidden_dim = glove_dict['the'].shape[-1]
        mean_value = np.mean(list(glove_dict.values()))
        variance_value = np.var(list(glove_dict.values()))
        left_boundary = mean_value - 3 * np.sqrt(variance_value)
        right_boundary = mean_value + 3 * np.sqrt(variance_value)
        for key in glove_dict.keys():
            temp_clip = np.clip(glove_dict[key], left_boundary, right_boundary)
            temp = (temp_clip - mean_value) / (3 * np.sqrt(variance_value))
            glove_dict[key] = (temp + 1) / 2
        glove_dict = glove_dict
        glove_dict['<pad>'] = [0] * hidden_dim
        glove_dict['<unk>'] = [0] * hidden_dim
        return glove_dict
    
    glove_dict = get_dict()

    def one_zero_normal(text_list, dict):
        batch_embedding_list = []
        for text in text_list:
            text = text.lower()
            text_embedding = []
            words = list(map(lambda x: x if x in dict.keys() else '<unk>', text.strip().split()))
            if len(words) > args.sentence_length:
                words = words[:args.sentence_length]
            elif len(words) < args.sentence_length:
                while len(words) < args.sentence_length:
                    words.append('<pad>')
            for i in range(len(words)):
                text_embedding.append(dict[words[i]])
            batch_embedding_list.append(text_embedding)
        return batch_embedding_list

    student_data = []
    for i, batch in enumerate(teacher_data_loader):
        student_data.append(one_zero_normal(list(batch[0]), glove_dict))
    
    output_message("Distill Begins...")

    for epoch in tqdm(range(args.distill_epoch)):
        teacher_model.eval()
        student_model.train()
        for i, batch in enumerate(teacher_data_loader):
            teacher_inputs = teacher_tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")
            to_device(teacher_inputs, args.device)
            teacher_outputs = teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
            teacher_hidden_states = teacher_outputs.hidden_states
            student_inputs = torch.tensor(np.array(student_data[i], dtype=float), dtype=float).to(args.device)
            if args.student_model_name == "dpcnn":
                student_hidden_states, student_logits = student_model(student_inputs)
                # print(f"teacher_hidden_states shape:{teacher_hidden_states[-1][:, 0].shape}") # batch * 768
                teacher_embed = teacher_hidden_states[-1][:, 0]
                student_embed = student_hidden_states
                # print(f"student_hidden_states shape:{student_hidden_states.shape}") # batch * fitter_num
                embed_loss = F.mse_loss(student_embed, teacher_embed)
                logit_loss = F.mse_loss(student_logits, teacher_logits)
                loss = args.logit_loss_weight * logit_loss + args.feature_loss_weight * embed_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                student_logits = student_model(student_inputs)
                logit_loss = F.mse_loss(student_logits, teacher_logits)
                loss = logit_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        saved_path = FileCreater.build_saving_file(args, description="-epoch{}".format(epoch))
        save_model_to_file(save_path=saved_path, model=student_model)

        student_model.eval()
        with torch.no_grad():
            correct = 0
            for data, y_batch in args.test_dataloader:
                data = data.to(args.device)
                y_batch = y_batch.to(args.device)
                if args.student_model_name == "dpcnn":
                    _, output = student_model(data)
                else:
                    output = student_model(data)
                correct += int(y_batch.eq(torch.max(output,1)[1]).sum())
            output_message(f"Epoch {epoch} Acc: {float(correct/len(test_dataset))}")

            # dev_correct = 0
            # for data, y_batch in args.dev_dataloader:
            #     data = data.to(args.device)
            #     y_batch = y_batch.to(args.device)
            #     if args.student_model_name == "dpcnn":
            #         _, output = student_model(data)
            #     else:
            #         output = student_model(data)
            #     dev_correct += int(y_batch.eq(torch.max(output,1)[1]).sum())
            # output_message(f"Epoch {epoch} Acc: {float(dev_correct/len(dev_dataset))}")


if __name__ == "__main__":
    args = SNNArgs.parse()
    build_environment(args)
    FileCreater.build_directory(args, args.logging_dir, 'logging', args.args_for_logging)
    FileCreater.build_directory(args, args.saving_dir, 'saving', args.args_for_logging)
    FileCreater.build_logging(args)
    output_message("Program args: {}".format(args))
    if args.mode == 'train' and args.model_mode == "snn":
        train(args)
    elif args.mode == 'train' and args.model_mode == "ann":
        ann_train(args)
    elif args.mode == 'attack':
        attack(args)
    elif args.mode == "conversion":
        conversion(args)
    elif args.mode == "distill":
        distill(args)