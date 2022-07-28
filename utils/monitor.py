import torch
from utils.public import output_message



class Monitor:

    MONITOR_TENSOR_DICT = {}
    TOTAL_COUNT = {}
    _EPOCH = 0

    def __init__(self) -> None:
        self.monitor_tensor_list = None
        pass

    @staticmethod
    def create_epoch_monitor():
        Monitor.MONITOR_TENSOR_DICT[Monitor._EPOCH] = []
        Monitor.TOTAL_COUNT[Monitor._EPOCH] = 0

    @staticmethod
    def append_monitor(tensor):
        if Monitor._EPOCH not in Monitor.MONITOR_TENSOR_DICT.keys():
            Monitor.create_epoch_monitor()
        t_clone = tensor.clone().detach()
        Monitor.MONITOR_TENSOR_DICT[Monitor._EPOCH].append(t_clone)

    @staticmethod
    def add_monitor(tensor, position):
        assert Monitor._EPOCH in Monitor.MONITOR_TENSOR_DICT.keys()
        if position > len(Monitor.MONITOR_TENSOR_DICT[Monitor._EPOCH]):
            raise Exception("Please add tensor in right pos, from 0 to n.")
        elif position == len(Monitor.MONITOR_TENSOR_DICT[Monitor._EPOCH]):
            Monitor.append_monitor(tensor)
        else:
            t_clone = tensor.clone().detach()
            if tensor.shape == Monitor.MONITOR_TENSOR_DICT[Monitor._EPOCH][position].shape:
                Monitor.MONITOR_TENSOR_DICT[Monitor._EPOCH][position] += t_clone
            else:
                raise Exception("Wrong input shape")
        if position == 0:
            Monitor.TOTAL_COUNT[Monitor._EPOCH] += 1

    @staticmethod
    def calculate_dead_ratio_by_epoch(epoch):
        ratio_list = []
        for pos in range(len(Monitor.MONITOR_TENSOR_DICT[epoch])):
            zero_mask = torch.eq(Monitor.MONITOR_TENSOR_DICT[epoch][pos], 0)
            zero_count = torch.sum(zero_mask)
            shape_list = Monitor.MONITOR_TENSOR_DICT[epoch][pos].shape
            neuron_num = 1
            for s in shape_list:
                neuron_num *= s
            ratio_list.append((zero_count / neuron_num))
        return ratio_list

    @staticmethod
    def print_results_by_epoch(epoch):
        r_list = Monitor.calculate_dead_ratio_by_epoch(epoch)
        ratio_str = ", ".join(["spike no.{}: dead ratio:{}".format(index+1, value) 
            for index, value in enumerate(r_list)])
        output_message("Epoch {} ".format(epoch) + ratio_str)

