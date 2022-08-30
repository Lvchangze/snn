from email.policy import strict
import os
import torch
import numpy as np
import random
import logging

def check_and_create_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def output_message(message: str):
    print(message)
    logging.info(message)

def save_model_to_file(save_path, model):
    if save_path[-4:] != '.pth':
        save_path = '{}.pth'.format(save_path)
    torch.save(model.state_dict(), save_path)
    logging.info('Saved model to {}'.format(save_path))
    logging.info("The number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters())))

def load_model_from_file(save_path, model):
    if save_path[-4:] != '.pth':
        save_path = '{}.pth'.format(save_path)
    model.load_state_dict(torch.load(save_path), strict=False)
    return

def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True