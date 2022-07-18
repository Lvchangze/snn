import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from snntorch import spikegen

sent_length_list = []

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

with open("data/sst2/train.txt", "r") as f:
    for line in f.readlines():
        temp = line.split('\t')
        sentence = temp[0].strip()
        sent_length_list.append(len(sentence.split()))

print("max length: ", np.max(sent_length_list))
print("min length: ", np.min(sent_length_list))
print("mean length: ", np.mean(sent_length_list))

start = 0  # 区间左端点
length = 5  # 区间长度
number_of_interval = 11  # 区间个数
intervals = {'{:.5f}~{:.5f}'.format(length * x + start, length * (x+1) + start): 0 for x in range(number_of_interval)}
interval_statistics(sent_length_list, intervals)