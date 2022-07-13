python preprocess.py \
    --vocab_path data/glove.6B.100d.txt \
    --dataset_name sst2 \
    --data_path data/sst2/train.txt \
    --sent_length 30 \
    --embedding_dim 100