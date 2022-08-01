import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from sklearn.metrics import f1_score
import random
import numpy as np
import json
from tqdm import tqdm
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size_train', type=int, default=2, help='batch size')
parser.add_argument('--batch_size_test', type=int, default=2, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=3e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='legal_bert_criminal', help='PLM')
parser.add_argument('--checkpoint', type=str, default="./weights/baselines", help='checkpoint path')
parser.add_argument('--bert_maxlen', type=int, default=512, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
parser.add_argument('--seed', type=int, default=1, help='random_seed')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='whether train')
parser.add_argument("--train", action="store_false", help="whether train")
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
data_path = "/competition_data/competition_stage_1_train.json"
data_path_test = "/competition_data/competition_stage_1_test.json"
data_path_pred = "/competition_data/submission.txt"
pretrained_bert_legal_criminal_fold = "/pretrain_model/bert_legal_criminal/"

def load_data(filename):
    """加载数据
    返回：[{...}]
    """
    all_data = []
    with open(filename) as f:
        for l in f:
            all_data.append(json.loads(l))
    return all_data

def load_data_test(filename):
    """加载数据
    返回：[{...}]
    """
    all_data = []
    ids = []
    with open(filename) as f:
        for l in f:
            item = json.loads(l)
            item['label'] = 0  # padding 0 (every int value is ok)
            ids.append(item['id'])
            all_data.append(item)
    return all_data, ids

def data_split(data, mode, splite_ratio=0.8):
    """划分训练集和验证集
    """
    splite_point1 = int(splite_ratio*len(data))
    if mode == 'train':
        D = data[:splite_point1]
    elif mode == 'valid':
        D = data[splite_point1:]
    else:
        print("mode type can only in train test valid")


    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D

def load_checkpoint(model, file_name=None):
    save_params = torch.load(file_name, map_location=device)
    model.load_state_dict(save_params["model"])


def save_checkpoint(model, optimizer, filename):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(save_params, filename)


class EarlyStop:
    def __init__(self, patience, max_or_min="max"):
        self.patience = patience
        self.best_value = 0.0
        self.best_epoch = 0
        self.max_or_min = max_or_min
    def step(self, current_value, current_epoch):
        if self.max_or_min == 'max':
            print("Current:{} Best:{}".format(current_value, self.best_value))
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_epoch = current_epoch
                return True
            return False
        elif self.max_or_min == 'min':
            print("Current:{} Best:{}".format(current_value, self.best_value))
            if current_value < self.best_value:
                self.best_value = current_value
                self.best_epoch = current_epoch
                return True
            return False
        else:
            print("early stop type is max or min")
            exit(-1)
    def stop_training(self, current_epoch) -> bool:
        return current_epoch - self.best_epoch > self.patience


class Dataset_Instances(Dataset):
    def __init__(self, data):
        super(Dataset_Instances, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return "".join(self.data[index]["Case_A"]), "".join(self.data[index]["Case_B"]), self.data[index]['label']

class Collate:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_criminal_fold)
        self.max_seq_len = args.bert_maxlen


    def __call__(self, batch):
        text_a, text_b, labels = [], [], []
        for item in batch:
            text_a.append(item[0])
            text_b.append(item[1])
            labels.append(item[2])
        dic_data_a = self.tokenizer.batch_encode_plus(text_a, padding=True, truncation=True,
                                                      max_length=self.max_seq_len, return_tensors='pt')
        dic_data_b = self.tokenizer.batch_encode_plus(text_b, padding=True, truncation=True,
                                                     max_length=self.max_seq_len, return_tensors='pt')

        return dic_data_a, dic_data_b, torch.tensor(labels)


def build_dataloader(data, batch_size, shuffle=True, num_workers=4):
    data_generator = Dataset_Instances(data)
    collate = Collate()
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )


class PredictorModel(nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_criminal_fold)
        self.model = BertModel.from_pretrained(pretrained_bert_legal_criminal_fold)
        self.configuration = self.model.config
        self.FFN = nn.Sequential(
            nn.Linear(3*self.configuration.hidden_size, self.configuration.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(self.configuration.hidden_size//2, 3),
            )


    def forward(self, text_a, text_b):
        output_text_a = self.model(**text_a).pooler_output
        output_text_b = self.model(**text_b).pooler_output
        data_p = torch.cat([output_text_a, output_text_b, torch.abs(output_text_a-output_text_b)], dim=-1)
        output = self.FFN(data_p)
        return output

def train_valid(model, train_dataloader, valid_dataloader, test_dataloader, ids):
    criterion = nn.CrossEntropyLoss()
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in list(model.named_parameters())], 'weight_decay_rate': args.weight_decay, 'lr': args.lr})
    optimizer = torch.optim.Adam(params=optimizer_grouped_parameters)
    early_stop = EarlyStop(args.early_stopping_patience)
    all_step_cnt = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        current_step = 0
        model.train()
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train',  ncols=100)
        for batch_data in pbar:
            all_step_cnt += 1
            Case_a, Case_b, label_batch = batch_data
            text_batch_a = Case_a.to(device)
            text_batch_b = Case_b.to(device)
            label_batch = label_batch.to(device)
            scores = model(text_batch_a, text_batch_b)
            loss = criterion(scores, label_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1
            pbar.set_description("train loss {} ".format(epoch_loss / current_step))
        epoch_loss = epoch_loss / current_step
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss: {:.4f} '.format(time_str, epoch, epoch_loss))
        model.eval()
        current_val_metric_value = evaluation(valid_dataloader, model, epoch)
        is_save = early_stop.step(current_val_metric_value, epoch)
        if is_save:
            test(test_dataloader, model, ids)
        else:
            pass
        if early_stop.stop_training(epoch):
            print("early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))

            break



def save_rationale(prediction_label, ids):
    """
    load test ids
    """

    relations = []
    selected_id_a = [[0, 1, 2] for _ in range(len(prediction_label))]
    selected_id_b = [[0, 1, 2] for _ in range(len(prediction_label))]
    for label in prediction_label:
        if label == 1 or label == 2:
            relations.append([[0, 0], [1, 1], [2, 2]])
        else:
            relations.append([])
    data_pred = {"id": ids, 'Case_A_rationales': selected_id_a, 'Case_B_rationales': selected_id_b,
                  "relation":relations, "label": prediction_label}

    pd_data_pred = pd.DataFrame(data_pred)
    pd_data_pred.to_csv(data_path_pred, index=False, sep='\t')


def evaluation(valid_dataloader, model, epoch):
    with torch.no_grad():
        current_step = 0
        all_true_label = []
        all_prediction_label = []
        pbar = tqdm(valid_dataloader, desc="Iteration")
        for batch_data in pbar:
            Case_a, Case_b, label_batch = batch_data
            text_batch_a = Case_a.to(device)
            text_batch_b = Case_b.to(device)
            label_batch = label_batch.to(device)
            scores = model(text_batch_a, text_batch_b)
            prediction_label = torch.argmax(scores, dim=-1).to(torch.long)
            all_true_label += label_batch.cpu().tolist()
            all_prediction_label += prediction_label.cpu().tolist()
            f1_batch = f1_score(all_true_label, all_prediction_label, average='macro')
            pbar.set_description("epoch {} f1 macro {}".format(epoch, f1_batch))
            current_step += 1
        f1_macro = f1_score(all_true_label, all_prediction_label, average='macro')
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} epoch {} f1 {} '.format(time_str, epoch, f1_macro))
    return f1_macro


def test(test_dataloader, model, ids):
    with torch.no_grad():
        all_prediction_label = []
        pbar = tqdm(test_dataloader, desc="Iteration")
        for batch_data in pbar:
            Case_a, Case_b, _ = batch_data
            text_batch_a = Case_a.to(device)
            text_batch_b = Case_b.to(device)
            scores = model(text_batch_a, text_batch_b)
            prediction_label = torch.argmax(scores, dim=-1).to(torch.long)
            all_prediction_label += prediction_label.cpu().tolist()
        save_rationale(all_prediction_label, ids)

if __name__ == '__main__':
    if args.train:
        data = load_data(data_path)
        train_data = data_split(data, 'train', splite_ratio=0.9)
        valid_data = data_split(data, 'valid', splite_ratio=0.9)
        train_data_loader = build_dataloader(train_data, args.batch_size_train, shuffle=True)
        valid_data_loader = build_dataloader(valid_data, args.batch_size_train, shuffle=False, num_workers=1)
        test_data, ids = load_data_test(data_path_test)
        test_data_loader = build_dataloader(test_data, args.batch_size_test, shuffle=False, num_workers=1)
        P_model = PredictorModel()
        P_model = P_model.to(device)
        train_valid(P_model, train_data_loader, valid_data_loader, test_data_loader, ids)


