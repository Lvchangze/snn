import numpy as np
import argparse
import pickle

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
    max_value = np.max(np.array(list(glove_dict.values())))
    min_value = np.min(np.array(list(glove_dict.values())))
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
        # padding
        for i in range(args.sent_length):
            if i >= len(text_list):
                embedding = mean_value
            else:
                word = text_list[i]
                embedding = glove_dict[word] if word in glove_dict.keys() else mean_value
            sent_embedding[i] = embedding
        embedding_tuple_list.append((sent_embedding, label))
        
    # save dict
    with open('data/glove.{}d.dict'.format(args.embedding_dim), 'wb') as f:
        pickle.dump(glove_dict, f, -1)

    # read dict
    # with open('data/glove.100d.dict', 'rb') as f:
    #     tuple_dataset = pickle.load(f)
