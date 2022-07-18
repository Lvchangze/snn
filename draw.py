import numpy as np
import argparse
import pickle
import torch
from tqdm import tqdm
from dataset import TensorDataset

def get_dict(vocab_path):
    dict = {}
    with open(vocab_path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            dict[word] = vector
    return dict

def interval_statistics(data, intervals):
    if len(data) == 0:
        return
    for num in data:
        for interval in intervals:
            lr = tuple(interval.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals[interval] += 1
    for key, value in intervals.items():
        print("%10s" % key, end='')  # 借助 end=''可以不换行
        print("%10s" % value, end='')  # "%10s" 右对齐
        print('%16s' % '{:.5%}'.format(value * 1.0 / len(data)))

if __name__ == "__main__":
    glove_dict = get_dict("data/glove.6B.100d.txt")
    max_value = np.max(np.array(list(glove_dict.values())))
    print(max_value)
    min_value = np.min(np.array(list(glove_dict.values())))
    print(min_value)
    mean_value = np.mean(list(glove_dict.values()))
    print(mean_value)
    variance_value = np.var(list(glove_dict.values()))
    print(variance_value)
    left_boundary = mean_value - 3 * np.sqrt(variance_value)
    print(left_boundary)
    right_boundary = mean_value + 3 * np.sqrt(variance_value)
    print(right_boundary)
    all_num = []
    for value in np.array(list(glove_dict.values())):
        for d in value:
            all_num.append(d)
    start = mean_value - 10 * np.sqrt(variance_value)  # 区间左端点
    length = np.sqrt(variance_value)  # 区间长度
    number_of_interval = 20  # 区间个数
    intervals = {'{:.5f}~{:.5f}'.format(length * x + start, length * (x+1) + start): 0 for x in range(number_of_interval)}
    interval_statistics(all_num, intervals)
