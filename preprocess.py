import numpy as np
import argparse
import pickle
import torch

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_path",
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        default="sst2",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--sent_length",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--embedding_dim",
        default=100,
        type=int,
    )
    

    args = parser.parse_args()
    glove_dict = get_dict(args.vocab_path)
    max_value = np.array([float(np.max(np.array(list(glove_dict.values()))))] * args.embedding_dim)
    min_value = np.array([float(np.min(np.array(list(glove_dict.values()))))] * args.embedding_dim)
    mean_value = np.mean(np.array(list(glove_dict.values())), axis=0)

    sample_list = []
    with open(args.data_path, "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sample_list.append((sentence, label))

    embedding_tuple_list = []
    for i in range(len(sample_list)):
        sent_embedding = np.array([[0] * args.embedding_dim] * args.sent_length, dtype=float)
        text_list = sample_list[i][0].split()
        label = sample_list[i][1]
        for i in range(args.sent_length):
            if i >= len(text_list):
                embedding = np.array([0] * args.embedding_dim, dtype=float) # zero padding
            else:
                word = text_list[i]
                embedding = glove_dict[word] if word in glove_dict.keys() else mean_value
                embedding = (embedding - min_value) / (max_value - min_value) # normalize
            sent_embedding[i] = embedding
        # print(i, sent_embedding)
        embedding_tuple_list.append((torch.tensor(sent_embedding), label))
    
    dataset = TensorDataset(embedding_tuple_list)
    # print(dataset[0])

    # save dataset
    with open('data/sst_2glove{}d.tensor_dataset'.format(args.embedding_dim), 'wb') as f:
        pickle.dump(dataset, f, -1)

