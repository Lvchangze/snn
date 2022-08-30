from ast import arg
import os
from typing import List
from utils.public import check_and_create_path
import time
import logging

class FileCreater():
    def __init__(self, args) -> None:
        self.args = args
        pass

    @staticmethod
    def build_directory(args, dir_name, style='logging', terms:List=None):
        temp_path = os.path.join(dir_name, "{}_{}".format(args.dataset_name, args.model_type))
        if terms:
            term_pairs = {}
            for item in terms:
                if not hasattr(args, item):
                    raise Exception("The term {} in args_for_logging is not set in SNNArgs".format(item))
                term_pairs[item] = getattr(args, item)
            path = "-".join(["{}{}".format(key, value) for key, value in term_pairs.items()])
            # for i in range(1, len(terms)):
            #     path += "_{}".format(terms[i])
            temp_path = os.path.join(dir_name, path)
        check_and_create_path(temp_path)
        if style == 'logging':
            args.logging_dir = temp_path
        elif style == 'saving':
            args.saving_dir = temp_path
        return temp_path

    @staticmethod
    def build_logging(args):
        local_time = time.localtime()
        file_name = "{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", local_time))
        args.logging_path = os.path.join(args.logging_dir, file_name)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=args.logging_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

    @staticmethod
    def build_saving_file(args, description=None):
        local_time = time.localtime()
        file_name = "{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", local_time))
        saving_path = os.path.join(args.saving_dir, file_name)
        if description:
            return "{}-{}".format(saving_path, description)
        return saving_path

        