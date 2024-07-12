# Spiking Convolutional Neural Networks for Text Classification (ICLR 2023, Poster)

https://openreview.net/forum?id=pgU3k7QXuz0

## Install Requirements

```shell
conda create -n snn python=3.7
conda activate snn
pip install -r requirements.txt
pip install -r textattack_r.txt
```

## Shift Pre-trained Word Embeddings to [0, 1]

```shell
cd data_preprocess
python tensor_encoder.py
python chinese_tensor_encoder.py
```

## Train Tailored Models

```shell
python main.py \
--mode train 
--model_mode ann 
--model_type textcnn
```

## Embed (Prepare for Random Initial Embedding)

## Conversion + Normalization methods

```shell
python main.py \
--mode conversion
--model_mode snn
--model_type textcnn
--conversion_mode normalize
--conversion_normalize_type model_base or data_base
```

## Conversion + Fine-tuning SNNs

```shell
python main.py \
--mode conversion
--model_mode snn
--model_type textcnn
--conversion_mode tune
```

## If Random Initial Embedding (without Pretrain)

### Prepare Sentence2index

```shell
cd data_preprocess
python sent2id.py
```

### Apply Embedding Layer before fed into SNNs

```shell
cd data_preprocess
python snn_wopretrain_tensor_encoder.py
```

## Notes

- Please prepare dateset files and pre-trained word embedding in yourself.
- Shell commands above only show how to run the program in different modes. Detailed hyper-parameters can be set as you want.
- Set parameter **random_tensor** to **True** when doing conversion + Normalization or fine-tune if you use random initial embedding.
- Some large files are uploaded here: link: https://pan.baidu.com/s/19l81POzDUj4mBAmncCGkiA, code: cehb
