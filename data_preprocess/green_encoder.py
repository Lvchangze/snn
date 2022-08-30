import copy
import torch
import os

def permu(bit, dep, a, ret):
    if (dep == bit):
        ret.append(copy.deepcopy(a))
        return
    a[dep] = 1
    permu(bit, dep+1, a, ret)
    a[dep] = 0
    permu(bit, dep+1, a, ret)

def green(bit, dep, res, a, ret):
    if (res == 0):
        ret.append(copy.deepcopy(a))
        return
    for i in range(bit-res, dep-1, -1):
        a[i] = 0
        green(bit, i+1, res-1, a, ret)
        a[i] = 1

def preprocess(bit=8):
    a = [0 for _ in range(bit)]
    permu_a, greencode = [], []
    permu(bit, 0, a, permu_a)
    a = [1 for _ in range(bit)]
    for num0 in range(bit+1):
        green(bit, 0, num0, a, greencode)
    permu_a = permu_a[::-1]
    greencode = greencode[::-1]
    return permu_a, greencode

def encode(x, codebook, step_num=32, bit=8):
    bin_num = 1 << bit
    stride = 1. / bin_num
    bin_idx = min(int(x / stride), bin_num-1)
    spike = codebook[bin_idx]
    return int(step_num / bit) * spike

def code_generate(input, num_steps, codebook):
    bit = codebook.shape[-1]
    repeat_num = num_steps // bit
    bin_num = 1 << bit
    stride = 1. / bin_num
    bin_idx = torch.clamp(torch.floor(input / stride), max=bin_num-1).int()
    output = torch.index_select(codebook, 0, bin_idx)
    repeat_shape = [1 for _ in range(len(output.shape))]
    repeat_shape[-1] = repeat_num
    output = output.repeat(repeat_shape)
    return output

class GreenEncoder():
    def __init__(self, args) -> None:
        self.bit = args.bit
        self.num_steps = args.num_steps
        self.data_dir = args.data_dir
        self.codebook_type = args.codebook_type
        pass

    def write_codebook(self):
        file_path = os.path.join(self.data_dir, 
            "codebook_type{}_bit{}.pth".format(self.codebook_type, self.bit))
        if os.path.exists(file_path):
            self.codebook = torch.load(file_path)
            return
        books = preprocess(bit=self.bit)
        if self.codebook_type == 'normal':
            codebook = books[0]
        elif self.codebook_type == 'green':
            codebook = books[1]
        else:
            raise Exception("No this codebook type!")
        torch_codebook = torch.tensor(codebook, dtype=torch.float)
        torch.save(torch_codebook, file_path)
        self.codebook = torch_codebook
        return

    def spike_gen(self, input, num_step):
        repeat_num = self.num_steps // self.bit
        if self.num_steps % self.bit:
            raise Warning("The num steps and bit is not suitable.")
        bin_num = 1 << self.bit
        stride = 1. / bin_num
        bin_idx = torch.clamp(torch.floor(input / stride), max=bin_num-1).int()
        origin_shape = bin_idx.shape
        flatten_bin_idx = bin_idx.view(-1)
        output = torch.index_select(self.codebook, 0, flatten_bin_idx)
        output = output.view(list(origin_shape)+[-1])
        repeat_shape = [1 for _ in range(len(output.shape))]
        repeat_shape[-1] = repeat_num
        output = output.repeat(repeat_shape)
        output = output.permute(2, 0, 1)
        return output


if __name__ == "__main__":
    permu_a, greencode = preprocess(4)
    print(encode(0.77, permu_a, step_num=16, bit=4))