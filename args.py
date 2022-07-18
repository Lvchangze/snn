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
        self.args_for_logging = ['optimizer_name','learning_rate', 'batch_size', 'epochs', 'beta']
        
        # training details
        self.mode = "train"
        self.dataset_name = 'sst2'
        self.label_num = 2
        self.seed = 42
        self.epochs = 10
        self.batch_size = 32
        self.sentence_length = 30
        self.hidden_dim = 100
        self.num_steps = 10
        self.loss = 'cross_entropy'
        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        self.optimizer_name = "Adamw"

        # file saver
        # please modify the renew function together
        self.data_path = "data/sst2/u_3v_sst_2_glove100d.tensor_dataset"
        self.test_data_path = "data/sst2/test_u_3v_sst_2_glove100d.tensor_dataset"
        self.workspace = '/home/xujh/snn'
        self.data_dir = os.path.join(self.workspace, "data", self.dataset_name)
        self.logging_dir = os.path.join(self.workspace, 'logs')
        self.saving_dir = os.path.join(self.workspace, "saved_models")

        
        # network details
        self.surrogate = 'fast_sigmoid'
        self.beta = 0.95
        self.model_type = 'textcnn'
        self.filters = [3,4,5]
        self.filter_num = 100

    def renew_args(self):
        self.data_dir = os.path.join(self.workspace, "data", self.dataset_name)
        self.logging_dir = os.path.join(self.workspace, 'logs')
        self.saving_dir = os.path.join(self.workspace, "saved_models")

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