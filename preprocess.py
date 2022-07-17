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
    max_vevtor = np.array([float(np.max(np.array(list(glove_dict.values()))))] * args.embedding_dim)
    min_vector = np.array([float(np.min(np.array(list(glove_dict.values()))))] * args.embedding_dim)
    mean_embedding = np.mean(np.array(list(glove_dict.values())), axis=0)
    mean_value = np.mean(list(glove_dict.values()))
    variance_value = np.var(list(glove_dict.values()))
    left_boundary = mean_value - 3 * np.sqrt(variance_value)
    print(left_boundary)
    right_boundary = mean_value + 3 * np.sqrt(variance_value)
    print(right_boundary)

    sample_list = []
    with open(args.data_path, "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sample_list.append((sentence, label))

    embedding_tuple_list = []
    for i in tqdm(range(len(sample_list))):
        sent_embedding = np.array([[0] * args.embedding_dim] * args.sent_length, dtype=float)
        text_list = sample_list[i][0].split()
        label = sample_list[i][1]
        for j in range(args.sent_length):
            if j >= len(text_list):
                embedding_norm = np.array([0] * args.embedding_dim, dtype=float) # zero padding
            else:
                word = text_list[j]
                embedding = glove_dict[word] if word in glove_dict.keys() else mean_embedding
                # N(0, 1)
                embedding = (embedding - np.array([mean_value] * args.embedding_dim)) / np.array([np.sqrt(variance_value)] * args.embedding_dim)
                embedding_norm = np.array([0] * args.embedding_dim, dtype=float)
                for k in range(args.embedding_dim):
                    if embedding[k] < left_boundary:
                        embedding_norm[k] = left_boundary
                    elif embedding[k] > right_boundary:
                        embedding_norm[k] = right_boundary
                    else:
                        embedding_norm[k] = embedding[k]
                # add abs(left_embedding)
                embedding_norm = embedding_norm + np.array([np.abs(left_boundary)] * args.embedding_dim)
            sent_embedding[j] = embedding_norm
        # print(i, sent_embedding)
        embedding_tuple_list.append((torch.tensor(sent_embedding), label))
    
    dataset = TensorDataset(embedding_tuple_list)

    # save dataset
    with open('data/test_u_3v_sst_2_glove{}d.tensor_dataset'.format(args.embedding_dim), 'wb') as f:
        pickle.dump(dataset, f, -1)

