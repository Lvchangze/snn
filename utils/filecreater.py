import os
from typing import List
from utils.public import check_and_create_path

class FileCreater():
    def __init__(self, args) -> None:
        self.args = args
        pass

    @staticmethod
    def build_directory(dir_name, args, term:List=None):
        if not term or len(term) == 0:
            args.logging_path = os.path.join()
        pass