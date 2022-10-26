import argparse
from typing import Any
import os

class SNNArgs(argparse.Namespace):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # term for logging when doing tuning experiments
        # if you want to add some new args for logging
        # plz be careful to the changed saving and logging dirs
        # SUGGESTION: write exp with args_for_logging and save them in every manytask json file 

        # training details
        self.mode = "distill"  # ['train', 'attack', 'conversion', 'distill']
        self.model_mode = "snn"   # ['snn', 'ann']
        self.model_type = 'dpcnn'  # ["textcnn", "lstm", "dpcnn"]
        
        self.dataset_name = 'sst2'
        self.label_num = 2
        self.seed = 42
        self.use_seed = "False"
        self.epochs = 50
        self.batch_size = 32
        self.sentence_length = 25
        self.hidden_dim = 300
        self.num_steps = 50
        self.hidden_layer_num = 200  # hidden_layer_neuron_number
        self.loss = 'ce_rate'
        self.learning_rate = 1e-4
        self.weight_decay = 0.0
        self.dropout_p = 0.5
        self.optimizer_name = "Adamw"
        self.encode = "rate"  #['rate', 'latency']
        self.ensemble = "False"
        self.ensemble_class = 2
        
        self.max_len = 25
        self.attack_method = 'textfooler' # ["textfooler", "bae", "textbugger", "pso", "pwws", "deepwordbug"]
        self.attack_model_path = 'saved_models/best.pth'
        self.attack_times = 1
        self.attack_numbers = 1000
        self.attack_text_path = f"data/{self.dataset_name}/test.txt"
        self.neighbour_vocab_size = 15
        self.modify_ratio = 0.1
        # self.query_budget_size = 50
        self.sentence_similarity = 0.8

        # for codebook
        self.use_codebook = 'False'
        self.bit = 8
        self.codebook_type = 'green' # ['normal', 'green']
        
        # file saver
        # please modify the renew function together
        self.pretrain_embedding_name = "glove" # ['glove', 'word2vec']
        self.data_path = f"data/{self.dataset_name}/train_u_3v_{self.dataset_name}_{self.pretrain_embedding_name}{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.test_data_path = f"data/{self.dataset_name}/test_u_3v_{self.dataset_name}_{self.pretrain_embedding_name}{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.dev_data_path = f"data/{self.dataset_name}/dev_u_3v_{self.dataset_name}_{self.pretrain_embedding_name}{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.workspace = '/home/lvchangze/snn'
        self.data_dir = os.path.join(self.workspace, "data", self.dataset_name)
        self.logging_dir = os.path.join(self.workspace, 'logs')
        self.saving_dir = os.path.join(self.workspace, "saved_models")
        self.vocab_path = os.path.join(self.workspace, f"data/glove.6B.{self.hidden_dim}d.txt")
        self.attack_logging_dir = os.path.join(self.workspace, 'logs_attack')

        # network details
        self.surrogate = 'fast_sigmoid'
        self.beta = 1.0
        self.filters = [3,4,5]
        self.filter_num = 100
        self.initial_method = 'zero' # ['zero', 'normal', 'kaiming', 'xavier', 'k+n', 'k+x']
        self.positive_init_rate = 0.55
        self.threshold = 1.0

        # monitor
        self.dead_neuron_checker = "False"

        # conversion
        self.conversion_model_path = "saved_models/2022-09-17 22:53:22.log--epoch4.pth"
        self.conversion_mode = "normalize"              # ["tune", "normalize"]
        self.conversion_normalize_type = "model_base"   # ["model_base", "data_base"]

        # bilstm
        self.lstm_hidden_size = 150                     # [150, 300]
        self.lstm_fc1_num = 200                         # [200, 400]
        self.lstm_layers_num = 1
        self.bidirectional = "True"

        # distill
        self.teacher_model_path = "saved_models/bert-base-uncased_2022-09-14 18:27:35_epoch0_0.9335529928610653"
        self.logit_loss_weight = 1.0
        self.feature_loss_weight = 10
        self.student_model_name = "dpcnn"
        self.distill_batch = 32
        self.distill_epoch = 30
        self.data_augment_path = f"data/{self.dataset_name}/train_augment.txt"

        # dpcnn
        self.dpcnn_block_num = 2
        self.dpcnn_step_length = 5


    def renew_args(self):
        if self.model_mode == "ann" and self.mode == "train":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "sentence_length", "dropout_p", "weight_decay", "batch_size", "learning_rate", "label_num"]
        elif self.mode == "attack":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "attack_method","attack_times","attack_numbers"]
        elif self.model_mode == "snn" and self.mode == "train":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "label_num", "positive_init_rate", 'num_steps', 'learning_rate', "label_num"]
        elif self.mode == "conversion" and self.conversion_mode == "normalize":
            self.args_for_logging = ["model_mode", "mode", "conversion_mode", "model_type", "dataset_name", 'conversion_normalize_type']
        elif self.mode == "conversion" and self.conversion_mode == "tune":
            self.args_for_logging = ["model_mode", "mode", "conversion_mode", "model_type", "dataset_name", "conversion_normalize_type", "label_num", "positive_init_rate", 'num_steps', 'learning_rate']
        elif self.mode == "distill":
            self.args_for_logging = ["model_mode", "mode", "student_model_name", "dataset_name", "label_num", "logit_loss_weight","feature_loss_weight" ,"student_model_name", "distill_batch", "distill_epoch"]
        self.data_dir = os.path.join(self.workspace, "data", self.dataset_name)
        self.logging_dir = os.path.join(self.workspace, 'logs')
        self.saving_dir = os.path.join(self.workspace, "saved_models")
        self.vocab_path = os.path.join(self.workspace, f"data/glove.6B.{self.hidden_dim}d.txt")
        self.data_path = f"data/{self.dataset_name}/train_u_3v_{self.dataset_name}_{self.pretrain_embedding_name}{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.test_data_path = f"data/{self.dataset_name}/test_u_3v_{self.dataset_name}_{self.pretrain_embedding_name}{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.dev_data_path = f"data/{self.dataset_name}/dev_u_3v_{self.dataset_name}_{self.pretrain_embedding_name}{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.data_augment_path = f"data/{self.dataset_name}/train_augment.txt"
        self.attack_text_path = f"data/{self.dataset_name}/test.txt"

    @staticmethod
    def parse(verbose=False):
        parser = argparse.ArgumentParser()
        default_args = SNNArgs()
        for k, v in default_args.__dict__.items():
            if type(v) == bool:
                raise Exception("please convert bool into str type")
            parser.add_argument('--{}'.format(k),
                    action='store',
                    default=v,
                    type=type(v),
                    dest=str(k))
        parsed_args, _ = parser.parse_known_args(namespace=default_args)
        parsed_args.renew_args()
        if verbose:
            print("Args:")
            for k, v in parsed_args.__dict__.items():
                print("\t--{}={}".format(k, v))
        return parsed_args