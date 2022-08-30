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

        self.args_for_logging = ["dataset_name", "label_num", "positive_init_rate", 'num_steps', 'learning_rate']
        
        # training details
        self.mode = "attack"  # ['train', 'attack']
        self.dataset_name = 'sst2'
        self.label_num = 2
        self.seed = 42
        self.use_seed = "True"
        self.epochs = 20
        self.batch_size = 32
        self.sentence_length = 25
        self.hidden_dim = 100
        self.num_steps = 50
        self.hidden_layer_num = 200  # hidden_layer_neuron_number
        self.loss = 'ce_rate'
        self.learning_rate = 1e-4
        self.weight_decay = 0
        self.optimizer_name = "Adamw"
        self.encode = "rate"  #['rate', 'latency']
        self.ensemble = "False"
        self.max_len = 25
        self.attack_method = 'textfooler' # ['textfooler', 'bae']
        self.attack_model_path = 'saved_models/test.pth'
        self.attack_times = 5
        self.attack_numbers = 1000

        # for codebook
        self.use_codebook = 'False'
        self.bit = 8
        self.codebook_type = 'green' # ['normal', 'green']
        
        # file saver
        # please modify the renew function together
        self.data_path = f"data/{self.dataset_name}/train_u_3v_{self.dataset_name}_glove100d_sent_len{self.sentence_length}.tensor_dataset"
        self.test_data_path = f"data/{self.dataset_name}/test_u_3v_{self.dataset_name}_glove100d_sent_len{self.sentence_length}.tensor_dataset"
        self.workspace = '/home/xujh/snn'
        self.data_dir = os.path.join(self.workspace, "data", self.dataset_name)
        self.logging_dir = os.path.join(self.workspace, 'logs')
        self.saving_dir = os.path.join(self.workspace, "saved_models")
        self.vocab_path = os.path.join(self.workspace, "data/glove.6B.100d.txt")
        self.attack_logging_dir = os.path.join(self.workspace, 'logs_attack')

        # network details
        self.surrogate = 'fast_sigmoid'
        self.beta = 0.95
        self.model_type = 'textcnn'
        self.filters = [3,4,5]
        self.filter_num = 100
        self.initial_method = 'zero' # ['zero', 'normal', 'kaiming', 'xavier', 'k+n', 'k+x']
        self.positive_init_rate = 0.55

        # monitor
        self.dead_neuron_checker = "False"

    def renew_args(self):
        self.data_dir = os.path.join(self.workspace, "data", self.dataset_name)
        self.logging_dir = os.path.join(self.workspace, 'logs')
        self.saving_dir = os.path.join(self.workspace, "saved_models")
        self.data_path = f"data/{self.dataset_name}/train_u_3v_{self.dataset_name}_glove100d_sent_len{self.sentence_length}.tensor_dataset"
        self.test_data_path = f"data/{self.dataset_name}/test_u_3v_{self.dataset_name}_glove100d_sent_len{self.sentence_length}.tensor_dataset"

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