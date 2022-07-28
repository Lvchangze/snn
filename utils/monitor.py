from utils.public import output_message

MONITOR_TENSOR_DICT = {}
TOTAL_COUNT = {}
_EPOCH = 0

class Monitor:
    def __init__(self) -> None:
        self.monitor_tensor_list = None
        pass

    @staticmethod
    def create_epoch_monitor():
        MONITOR_TENSOR_DICT[_EPOCH] = []
        TOTAL_COUNT[_EPOCH] = 0

    @staticmethod
    def append_monitor(tensor):
        if _EPOCH not in MONITOR_TENSOR_DICT.keys():
            Monitor.create_epoch_monitor()
        t_clone = tensor.clone().detach()
        MONITOR_TENSOR_DICT[_EPOCH].append(t_clone)
        TOTAL_COUNT[_EPOCH] += 1

    @staticmethod
    def add_monitor(tensor, position):
        assert _EPOCH in MONITOR_TENSOR_DICT.keys()
        if position > len(MONITOR_TENSOR_DICT[_EPOCH]):
            raise Exception("Please add tensor in right pos, from 0 to n.")
        elif position == len(MONITOR_TENSOR_DICT[_EPOCH]):
            Monitor.append_monitor(tensor)
        else:
            if tensor.shape == MONITOR_TENSOR_DICT[_EPOCH][position].shape:
                MONITOR_TENSOR_DICT[_EPOCH][position] += tensor
            else:
                raise Exception("Wrong input shape")
        if position == 0:
            TOTAL_COUNT[_EPOCH] += 1



    @staticmethod
    def print_results_by_epoch(epoch):
        output_message("Epoch {} ".format(epoch))

