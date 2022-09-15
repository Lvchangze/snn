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
        self.mode = "conversion"  # ['train', 'attack', 'conversion', 'distill']
        self.model_mode = "ann"   # ['snn', 'ann']
        self.model_type = 'lstm'  # ["textcnn", "lstm"]
        
        self.dataset_name = 'sst2'
        self.label_num = 2
        self.seed = 42
        self.use_seed = "False"
        self.epochs = 50
        self.batch_size = 32
        self.sentence_length = 25
        self.hidden_dim = 100
        self.num_steps = 50
        self.hidden_layer_num = 200  # hidden_layer_neuron_number
        self.loss = 'ce_rate'
        self.learning_rate = 1e-4
        self.weight_decay = 0.0
        self.dropout_p = 0.5
        self.optimizer_name = "Adamw"
        self.encode = "rate"  #['rate', 'latency']
        self.ensemble = "False"
        self.max_len = 25
        self.attack_method = 'textfooler' # ["textfooler", "bae", "textbugger", "pso", "pwws", "deepwordbug"]
        self.attack_model_path = 'saved_models/best.pth'
        self.attack_times = 5
        self.attack_numbers = 1000

        # for codebook
        self.use_codebook = 'False'
        self.bit = 8
        self.codebook_type = 'green' # ['normal', 'green']
        
        # file saver
        # please modify the renew function together
        self.data_path = f"data/{self.dataset_name}/train_u_3v_{self.dataset_name}_glove{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.test_data_path = f"data/{self.dataset_name}/test_u_3v_{self.dataset_name}_glove{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
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

        # monitor
        self.dead_neuron_checker = "False"

        # conversion
        self.conversion_model_path = "saved_models/conversion.pth"
        self.conversion_mode = "normalize"              # ["tune", "normalize"]
        self.conversion_normalize_type = "model_base"   # ["model_base", "data_base"]

        # bilstm
        self.lstm_hidden_size = 150                     # [150, 300]
        self.lstm_fc1_num = 200                         # [200, 400]
        self.lstm_layers_num = 1
        self.bidirectional = "True"

        # distill
        self.teacher_model_path = "saved_models/bert-base-uncased_2022-09-14 18:27:35_epoch0_0.9335529928610653"
        self.distill_loss_alpha = 0.0
        self.student_model_name = "lstm"
        self.distill_batch = 50
        self.distill_epoch = 30
        self.data_augment_path = f"data/{self.dataset_name}/train_augment.txt"


    def renew_args(self):
        if self.model_mode == "ann" and self.mode == "train":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "sentence_length", "dropout_p", "weight_decay", "batch_size", "learning_rate"]
        elif self.mode == "attack":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "attack_method","attack_times","attack_numbers"]
        elif self.model_mode == "snn" and self.mode == "train":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "label_num", "positive_init_rate", 'num_steps', 'learning_rate']
        elif self.mode == "conversion" and self.conversion_mode == "normalize":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", 'conversion_normalize_type']
        elif self.mode == "conversion" and self.conversion_mode == "tune":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "conversion_normalize_type", "label_num", "positive_init_rate", 'num_steps', 'learning_rate']
        elif self.mode == "distill":
            self.args_for_logging = ["model_mode", "mode", "model_type", "dataset_name", "label_num", "distill_loss_alpha", "student_model_name", "distill_batch", "distill_epoch"]
        self.data_dir = os.path.join(self.workspace, "data", self.dataset_name)
        self.logging_dir = os.path.join(self.workspace, 'logs')
        self.saving_dir = os.path.join(self.workspace, "saved_models")
        self.vocab_path = os.path.join(self.workspace, f"data/glove.6B.{self.hidden_dim}d.txt")
        self.data_path = f"data/{self.dataset_name}/train_u_3v_{self.dataset_name}_glove{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.test_data_path = f"data/{self.dataset_name}/test_u_3v_{self.dataset_name}_glove{self.hidden_dim}d_sent_len{self.sentence_length}.tensor_dataset"
        self.data_augment_path = f"data/{self.dataset_name}/train_augment.txt"

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