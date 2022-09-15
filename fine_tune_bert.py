import torch
import torch.nn as nn
import pickle
import argparse
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
from dataset import TxtDataset
from transformers import BertTokenizer, BertForSequenceClassification
from model import ANN_BiLSTM

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="sst2",
        type=str,
    )
    parser.add_argument(
        "--train_data_path",
        default="data/sst2/train.txt",
        type=str,
    )
    parser.add_argument(
        "--test_data_path",
        default="data/sst2/test.txt",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=50,
        type=int
    )
    parser.add_argument(
        "--fine_tune_lr",
        default=5e-5,
        type=float
    )
    parser.add_argument(
        "--dropout_p",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float
    )
    parser.add_argument(
        "--epochs",
        default=3,
        type=int
    )
    parser.add_argument(
        "--teacher_model_name",
        default="bert-base-uncased",
        type=str
    )
    parser.add_argument(
        "--teacher_model_path",
        default="saved_models/bert-base-uncased_2022-09-14 18:27:35_epoch0_0.9335529928610653",
        type=str
    )
    parser.add_argument(
        "--label_num",
        default=2,
        type=int
    )
    args = parser.parse_args()
    return args

def fine_tune_teacher_model(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_name)
    model = BertForSequenceClassification.from_pretrained(args.teacher_model_name, num_labels=args.label_num)
    optimizer = Adam(model.parameters(), lr=args.fine_tune_lr)
    train_data_loader = DataLoader(dataset=TxtDataset(data_path=args.train_data_path), batch_size= args.batch_size, shuffle=True)
    test_dataset = TxtDataset(data_path=args.test_data_path)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False)
    
    device_ids = [i for i in range(torch.cuda.device_count())]
    
    model.train()
    model.to(device)
    
    for epoch in tqdm(range(args.epochs)):
        for i, batch in enumerate(train_data_loader):
            inputs = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")
            labels = batch[1].to(device)
            to_device(inputs, device)
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        label = []
        with torch.no_grad():
            all = len(test_dataset)
            correct = 0
            model.eval()
            for batch in test_data_loader:
                b_y = batch[1]
                input_dict = tokenizer(batch[0], return_tensors='pt', padding=True, truncation=True)
                to_device(input_dict, "cuda")
                output = (model(**input_dict).logits).to("cpu")
                for line in torch.softmax(output, 1):
                    label.append(line.numpy())
                correct += int(b_y.eq(torch.max(output,1)[1]).sum())
        label = np.asarray(label)
        acc = float(correct)/all 
        print(f"Epoch {epoch} Acc: {float(correct/len(test_dataset))}")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        tokenizer.save_pretrained(f"saved_models/{args.teacher_model_name}_{current_time}_{args.dataset_name}_epoch{epoch}_{acc}")
        model.save_pretrained(f"saved_models/{args.teacher_model_name}_{current_time}_{args.dataset_name}_epoch{epoch}_{acc}")
        # model.module.save_pretrained(f"saved_models/{args.teacher_model_name}_{current_time}_{args.dataset_name}_epoch{epoch}_{acc}")
    pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _args = args()
    fine_tune_teacher_model(_args)