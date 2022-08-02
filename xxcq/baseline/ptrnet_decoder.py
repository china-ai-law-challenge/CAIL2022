import sys
import os
import numpy as np
import random

from collections import OrderedDict
import pickle
import datetime
import json
from tqdm import tqdm
from recordclass import recordclass  #提供tuple结构，占用内存少
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.deterministic = True

# @inproceedings{nayak2019ptrnetdecoding,
#   author    = {Nayak, Tapas and Ng, Hwee Tou},
#   title     = {Effective Modeling of Encoder-Decoder Architecture for Joint Entity and Relation Extraction},
#   booktitle = {Proceedings of The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
#   year      = {2020}
# }

def custom_print(*msg):  #print变量、写日志文件
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')  #写入日志文件中
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


def load_word_embedding(embed_file, vocab):  #载入词向量
    custom_print('vocab length:', len(vocab))
    embed_vocab = OrderedDict()
    embed_matrix = list()

    embed_vocab['<PAD>'] = 0
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))

    embed_vocab['<UNK>'] = 1
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

    word_idx = 2
    with open(embed_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab and vocab[word] >= word_min_freq:
                vec = [np.float32(val) for val in parts[1:]]
                embed_matrix.append(vec)
                embed_vocab[word] = word_idx
                word_idx += 1

    for word in vocab:
        if word not in embed_vocab and vocab[word] >= word_min_freq:
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            word_idx += 1

    custom_print('embed dictionary length:', len(embed_vocab))
    return embed_vocab, np.array(embed_matrix, dtype=np.float32)


def build_vocab(data, save_vocab, embedding_file):  #建词典 字符级、词级 输出vocal.pkl文件
    vocab = OrderedDict()
    char_v = OrderedDict()  #对字典中的元素进行排序
    char_v['<PAD>'] = 0
    char_v['<UNK>'] = 1
    char_idx = 2
    for d in data:
        for word in d.SrcWords:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

            for c in word:
                if c not in char_v:
                    char_v[c] = char_idx
                    char_idx += 1

    word_v, embed_matrix = load_word_embedding(embedding_file, vocab)
    output = open(save_vocab, 'wb')
    pickle.dump([word_v, char_v], output)
    output.close()
    return word_v, char_v, embed_matrix


def load_vocab(vocab_file):  #导入词典  获得字和词的词典
    with open(vocab_file, 'rb') as f:
        embed_vocab, char_vocab = pickle.load(f)
    return embed_vocab, char_vocab


def get_adj_mat(sent_len, amat):  #建邻接矩阵 这里文件是预处理好的 只对邻接矩阵文件做导入 应该是在GNN中需要用到
    K = 5
    adj_mat = np.zeros((sent_len, sent_len), np.float32)
    for i in range(len(amat)):
        for j in range(len(amat)):
            if 0 <= amat[i][j] <= K:
                adj_mat[i][j] = 1.0 / math.pow(2, amat[i][j])
            else:
                adj_mat[i][j] = 0

    for i in range(sent_len):
        adj_mat[i][i] = 1
    return adj_mat


def get_data(src_lines, trg_lines, adj_lines, datatype):#获取数据、将json格式的数据处理成需要的格式 存进samples数组中
    samples = []
    uid = 1
    for i in range(0, len(src_lines)):
        src_line = src_lines[i].strip()
        trg_line = trg_lines[i].strip()
        src_words = src_line.split()

        trg_rels = []
        trg_pointers = []
        parts = trg_line.split('|')
        if datatype == 1:
            random.shuffle(parts)

        adj_data = json.loads(adj_lines[i])
        adj_mat = get_adj_mat(len(src_words), adj_data['adj_mat'])

        for part in parts:
            elements = part.strip().split()
            trg_rels.append(relnameToIdx[elements[4]])
            trg_pointers.append((int(elements[0]), int(elements[1]), int(elements[2]), int(elements[3])))

        if datatype == 1 and (len(src_words) > max_src_len or len(trg_rels) > max_trg_len):
            continue

        sample = Sample(Id=uid, SrcLen=len(src_words), SrcWords=src_words, TrgLen=len(trg_rels), TrgRels=trg_rels,
                        TrgPointers=trg_pointers, AdjMat=adj_mat)
        
        samples.append(sample)
        uid += 1
    return samples


def read_data(src_file, trg_file, adj_file, datatype):  #从处理好的文档中读取数据、src是文本文件、trg是指针文件也就是结果对应的序列、adj是邻接矩阵、datatype表示训练开发和测试、训练集中数据需要随机打乱
    reader = open(src_file)
    src_lines = reader.readlines()
    reader.close()

    reader = open(trg_file)
    trg_lines = reader.readlines()
    reader.close()

    reader = open(adj_file)
    adj_lines = reader.readlines()
    reader.close()

    # l = 1000
    # src_lines = src_lines[0:min(l, len(src_lines))]
    # trg_lines = trg_lines[0:min(l, len(trg_lines))]
    # adj_lines = adj_lines[0:min(l, len(adj_lines))]

    data = get_data(src_lines, trg_lines, adj_lines, datatype)
    return data


def get_relations(file_name):  #对关系进行id表示 即输出关系的id表示
    nameToIdx = OrderedDict()
    idxToName = OrderedDict()
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    nameToIdx['<PAD>'] = 0
    idxToName[0] = '<PAD>'
    # nameToIdx['<SOS>'] = 1
    # idxToName[1] = '<SOS>'
    nameToIdx['None'] = 1
    idxToName[1] = 'None'
    idx = 2
    for line in lines:
        nameToIdx[line.strip()] = idx
        idxToName[idx] = line.strip()
        idx += 1
    # print(len(nameToIdx),len(nameToIdx),nameToIdx, idxToName)
    return nameToIdx, idxToName

#获得答案的序列 包括两个实体的开始和位置
def get_answer_pointers(arg1start_preds, arg1end_preds, arg2start_preds, arg2end_preds, sent_len):
    arg1_prob = -1.0
    arg1start = -1
    arg1end = -1
    max_ent_len = 5
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob:
                arg1_prob = arg1start_preds[i] * arg1end_preds[j]
                arg1start = i
                arg1end = j

    arg2_prob = -1.0
    arg2start = -1
    arg2end = -1
    for i in range(0, arg1start):
        for j in range(i, min(arg1start, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j
    for i in range(arg1end + 1, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j

    arg2_prob1 = -1.0
    arg2start1 = -1
    arg2end1 = -1
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob1:
                arg2_prob1 = arg2start_preds[i] * arg2end_preds[j]
                arg2start1 = i
                arg2end1 = j

    arg1_prob1 = -1.0
    arg1start1 = -1
    arg1end1 = -1
    for i in range(0, arg2start1):
        for j in range(i, min(arg2start1, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    for i in range(arg2end1 + 1, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    if arg1_prob * arg2_prob > arg1_prob1 * arg2_prob1:
        return arg1start, arg1end, arg2start, arg2end
    else:
        return arg1start1, arg1end1, arg2start1, arg2end1


def is_full_match(triplet, triplets):  #判断三元组triplet是否在triplets三元组列表中
    for t in triplets:
        if t[0] == triplet[0] and t[1] == triplet[1] and t[2] == triplet[2]:
            return True
    return False


def get_gt_triples(src_words, rels, pointers):  #获得真实的三元组
    triples = []
    i = 0
    for r in rels:
        print(src_words[0])
        arg1 = ' '.join(src_words[pointers[i][0]:pointers[i][1] + 1])
        arg2 = ' '.join(src_words[pointers[i][2]:pointers[i][3] + 1])
        triplet = (arg1.strip(), arg2.strip(), relIdxToName[r])
        if not is_full_match(triplet, triples):
            triples.append(triplet)
        i += 1
    return triples


def get_pred_triples(rel, arg1s, arg1e, arg2s, arg2e, src_words):  #获得当前语句中预测的三元组 返回无重复的三元组列表和有重复的三元列表
    triples = []
    all_triples = []
    for i in range(0, len(rel)):
        r = np.argmax(rel[i][1:]) + 1
        if r == relnameToIdx['None']:
            break
        s1, e1, s2, e2 = get_answer_pointers(arg1s[i], arg1e[i], arg2s[i], arg2e[i], len(src_words))
        arg1 = ' '.join(src_words[s1: e1 + 1])
        arg2 = ' '.join(src_words[s2: e2 + 1])
        arg1 = arg1.strip()
        arg2 = arg2.strip()
        if arg1 == arg2:  #如果预测的是实体是同一个就忽略
            continue
        triplet = (arg1, arg2, relIdxToName[r])
        all_triples.append(triplet)
        if not is_full_match(triplet, triples): #如果三元组在triples中，则不加入
            triples.append(triplet)
    return triples, all_triples  


def get_F1(data, preds): #计算评价指标  preds是预测的关系和实体位置，data是原文的samples 包含了一系列内容
    gt_pos = 0
    pred_pos = 0
    total_pred_pos = 0
    correct_pos = 0
    for i in range(0, len(data)):
        gt_triples = get_gt_triples(data[i].SrcWords, data[i].TrgRels, data[i].TrgPointers)
        # if i==0:  
        #     print('rel arg1s',preds[0][i].shape,preds[1][i].shape,preds[0][i],preds[1][i])
        pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                          preds[4][i], data[i].SrcWords)
        total_pred_pos += len(all_pred_triples)
        gt_pos += len(gt_triples)
        pred_pos += len(pred_triples)
        for gt_triple in gt_triples:
            if is_full_match(gt_triple, pred_triples):
                correct_pos += 1
    print('t:',total_pred_pos)
    return pred_pos, gt_pos, correct_pos


def write_test_res(data, preds, outfile):  #将预测结果写入输出文件中  preds是预测的关系和实体位置，data是原文的samples 包含了一系列内容
    writer = open(outfile, 'w')
    for i in range(0, len(data)):
        pred_triples, _ = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i], preds[4][i],
                                        data[i].SrcWords)
        pred_triples_str = []
        for pt in pred_triples:
            pred_triples_str.append(pt[0] + ' ; ' + pt[1] + ' ; ' + pt[2])
        writer.write(' | '.join(pred_triples_str) + '\n')
    writer.close()


def shuffle_data(data):  #按batch随机打乱数据
    custom_print(len(data))  #print变量、写日志文件
    data.sort(key=lambda x: x.SrcLen)
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_max_len(sample_batch):  #获得最长序列长度
    src_max_len = len(sample_batch[0].SrcWords)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].SrcWords) > src_max_len:
            src_max_len = len(sample_batch[idx].SrcWords)

    trg_max_len = len(sample_batch[0].TrgRels)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].TrgRels) > trg_max_len:
            trg_max_len = len(sample_batch[idx].TrgRels)

    return src_max_len, trg_max_len


def get_words_index_seq(words, max_len):
    seq = list()
    for word in words:
        if word in word_vocab:
            seq.append(word_vocab[word])
        else:
            seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        seq.append(word_vocab['<PAD>'])
    return seq


def get_char_seq(words, max_len):
    char_seq = list()
    for i in range(0, conv_filter_size - 1):
        char_seq.append(char_vocab['<PAD>'])
    for word in words:
        for c in word[0:min(len(word), max_word_len)]:
            if c in char_vocab:
                char_seq.append(char_vocab[c])
            else:
                char_seq.append(char_vocab['<UNK>'])
        pad_len = max_word_len - len(word)
        for i in range(0, pad_len):
            char_seq.append(char_vocab['<PAD>'])
        for i in range(0, conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])

    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        for i in range(0, max_word_len + conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])
    return char_seq


def get_relation_index_seq(rel_ids, max_len):
    seq = list()
    # seq.append(relnameToIdx['<SOS>'])
    for r in rel_ids:
        seq.append(r)
    seq.append(relnameToIdx['None'])
    pad_len = max_len + 1 - len(seq)
    for i in range(0, pad_len):
        seq.append(relnameToIdx['<PAD>'])
    return seq


def get_padded_pointers(pointers, pidx, max_len):
    idx_list = []
    for p in pointers:
        idx_list.append(p[pidx])
    pad_len = max_len + 1 - len(pointers)
    for i in range(0, pad_len):
        idx_list.append(-1)
    return idx_list


def get_padded_relations(rels, max_len):
    rel_list = []
    for r in rels:
        rel_list.append(r)
    rel_list.append(relnameToIdx['None'])
    pad_len = max_len + 1 - len(rel_list)
    for i in range(0, pad_len):
        rel_list.append(relnameToIdx['<PAD>'])
    return rel_list


def get_padded_mask(cur_len, max_len):
    mask_seq = list()
    for i in range(0, cur_len):
        mask_seq.append(0)
    pad_len = max_len - cur_len
    for i in range(0, pad_len):
        mask_seq.append(1)
    return mask_seq


def get_entity_masks(pointers, src_max, trg_max):
    arg1_masks = []
    arg2_masks = []
    for p in pointers:
        arg1_mask = [1 for i in range(src_max)]
        arg1_mask[p[0]] = 0
        arg1_mask[p[1]] = 0

        arg2_mask = [1 for i in range(src_max)]
        arg2_mask[p[2]] = 0
        arg2_mask[p[3]] = 0

        arg1_masks.append(arg1_mask)
        arg2_masks.append(arg2_mask)

    pad_len = trg_max + 1 -len(pointers)
    for i in range(0, pad_len):
        arg1_mask = [1 for i in range(src_max)]
        arg2_mask = [1 for i in range(src_max)]
        arg1_masks.append(arg1_mask)
        arg2_masks.append(arg2_mask)
    return arg1_masks, arg2_masks


def get_positional_index(sent_len, max_len):
    index_seq = [min(i + 1, max_positional_idx - 1) for i in range(sent_len)]
    index_seq += [0 for i in range(max_len - sent_len)]
    return index_seq


def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_src_max_len, batch_trg_max_len = get_max_len(cur_samples)
    batch_trg_max_len += 1
    src_words_list = list()
    src_words_mask_list = list()
    src_char_seq = list()
    decoder_input_list = list()
    adj_lst = []
    positional_index_list = []

    rel_seq = list()
    arg1_start_seq = list()
    arg1_end_seq = list()
    arg2_start_seq = list()
    arg2_end_seq = list()
    arg1_mask_seq = []
    arg2_mask_seq = []

    for sample in cur_samples:
        src_words_list.append(get_words_index_seq(sample.SrcWords, batch_src_max_len))
        src_words_mask_list.append(get_padded_mask(sample.SrcLen, batch_src_max_len))
        src_char_seq.append(get_char_seq(sample.SrcWords, batch_src_max_len))
        cur_masked_adj = np.zeros((batch_src_max_len, batch_src_max_len), dtype=np.float32)
        cur_masked_adj[:len(sample.SrcWords), :len(sample.SrcWords)] = sample.AdjMat
        adj_lst.append(cur_masked_adj)
        positional_index_list.append(get_positional_index(len(sample.SrcWords), batch_src_max_len))

        if is_training:
            arg1_start_seq.append(get_padded_pointers(sample.TrgPointers, 0, batch_trg_max_len))
            arg1_end_seq.append(get_padded_pointers(sample.TrgPointers, 1, batch_trg_max_len))
            arg2_start_seq.append(get_padded_pointers(sample.TrgPointers, 2, batch_trg_max_len))
            arg2_end_seq.append(get_padded_pointers(sample.TrgPointers, 3, batch_trg_max_len))
            rel_seq.append(get_padded_relations(sample.TrgRels, batch_trg_max_len))
            decoder_input_list.append(get_relation_index_seq(sample.TrgRels, batch_trg_max_len))

            arg1_mask, arg2_mask = get_entity_masks(sample.TrgPointers, batch_src_max_len, batch_trg_max_len)
            arg1_mask_seq.append(arg1_mask)
            arg2_mask_seq.append(arg2_mask)
        else:
            decoder_input_list.append(get_relation_index_seq([], 1))

    return {'src_words': np.array(src_words_list, dtype=np.float32),
            'positional_seq': np.array(positional_index_list),
            'src_words_mask': np.array(src_words_mask_list),
            'src_chars': np.array(src_char_seq),
            'decoder_input': np.array(decoder_input_list),
            'adj': np.array(adj_lst),
            'rel': np.array(rel_seq),
            'arg1_start':np.array(arg1_start_seq),
            'arg1_end': np.array(arg1_end_seq),
            'arg2_start': np.array(arg2_start_seq),
            'arg2_end': np.array(arg2_end_seq),
            'arg1_mask': np.array(arg1_mask_seq),
            'arg2_mask': np.array(arg2_mask_seq)}


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, pre_trained_embed_matrix, drop_out_rate):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(pre_trained_embed_matrix))
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        word_embeds = self.embeddings(words_seq)
        word_embeds = self.dropout(word_embeds)
        return word_embeds

    def weight(self):
        return self.embeddings.weight


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, head_cnt, h_dim):
        super(Multi_Head_Self_Attention, self).__init__()
        self.m = head_cnt
        self.hidden_dim = int(h_dim/self.m)
        self.q_head = nn.ModuleList()
        self.k_head = nn.ModuleList()
        self.v_head = nn.ModuleList()
        for i in range(self.m):
            self.q_head.append(nn.Linear(h_dim, self.hidden_dim))
            self.k_head.append(nn.Linear(h_dim, self.hidden_dim))
            self.v_head.append(nn.Linear(h_dim, self.hidden_dim))
        self.w = nn.Linear(h_dim, h_dim)
        self.w1 = nn.Linear(h_dim, h_dim)
        self.w2 = nn.Linear(h_dim, h_dim)

    def forward(self, Q, K, V):
        att = torch.bmm(self.q_head[0](Q), self.k_head[0](K).transpose(1, 2))
        att /= math.sqrt(self.hidden_dim)
        att = F.softmax(att, dim=-1)
        sent = torch.bmm(att, self.v_head[0](V))
        for i in range(1, self.m):
            att = torch.bmm(self.q_head[i](Q), self.k_head[i](K).transpose(1, 2))
            att /= math.sqrt(self.hidden_dim)
            att = F.softmax(att, dim=-1)
            cur_sent = torch.bmm(att, self.v_head[i](V))
            sent = torch.cat((sent, cur_sent), -1)
        sent = self.w(sent)
        sent = nn.LayerNorm(sent.size()[1:], elementwise_affine=False)(sent + Q)
        lin_sent = self.w2(nn.ReLU()(self.w1(sent)))
        sent = nn.LayerNorm(sent.size()[1:], elementwise_affine=False)(sent + lin_sent)
        return sent


class Multi_Head_Attentive_Sent(nn.Module):
    def __init__(self, head_cnt, h_dim):
        super(Multi_Head_Attentive_Sent, self).__init__()
        self.m = head_cnt
        self.hidden_dim = int(h_dim/self.m)
        self.q_head = nn.ModuleList()
        self.k_head = nn.ModuleList()
        self.v_head = nn.ModuleList()
        for i in range(self.m):
            self.q_head.append(nn.Linear(h_dim, self.hidden_dim))
            self.k_head.append(nn.Linear(h_dim, self.hidden_dim))
            self.v_head.append(nn.Linear(h_dim, self.hidden_dim))
        self.w = nn.Linear(h_dim, h_dim)

    def forward(self, enc_hs, arg, src_mask):
        att = torch.bmm(self.q_head[0](enc_hs), self.k_head[0](arg).unsqueeze(2)).squeeze()
        att /= math.sqrt(self.hidden_dim)
        att.data.masked_fill_(src_mask.data, -float('inf'))
        att = F.softmax(att, dim=-1)
        sent = torch.bmm(att.unsqueeze(1), self.v_head[0](enc_hs)).squeeze()
        for i in range(1, self.m):
            att = torch.bmm(self.q_head[i](enc_hs), self.k_head[i](arg).unsqueeze(2)).squeeze()
            att /= math.sqrt(self.hidden_dim)
            att.data.masked_fill_(src_mask.data, -float('inf'))
            att = F.softmax(att, dim=-1)
            cur_sent = torch.bmm(att.unsqueeze(1), self.v_head[i](enc_hs)).squeeze()
            sent = torch.cat((sent, cur_sent), -1)
        sent = self.w(sent)
        return sent


def multi_head_pooling(h, mask, pool_type='max'):
    if pool_type == 'max':
        h.data.masked_fill_(mask.unsqueeze(2).data, -float('inf'))
        pooled_h = torch.max(h, 1)[0]
    else:
        h.data.masked_fill_(mask.unsqueeze(2).data, 0)
        pooled_h = torch.max(h, 1)[0]
    return pooled_h


class Multi_Factor_Attention(nn.Module):
    def __init__(self, factor_cnt, in_dim, out_dim):
        self.drop_rate = drop_rate
        super(Multi_Factor_Attention, self).__init__()
        self.m = factor_cnt
        self.layers = nn.ModuleList()
        for i in range(self.m):
            self.layers.append(nn.Linear(in_dim, out_dim))
        self.w = nn.Linear(2 * in_dim, out_dim)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, enc_hs, src_mask, arg1, arg2):
        arg = torch.tanh(self.w(torch.cat((arg1, arg2), -1)))
        att = torch.bmm(torch.tanh(self.layers[0](enc_hs)), arg.unsqueeze(2)).squeeze()
        att.data.masked_fill_(src_mask.data, -float('inf'))
        att = F.softmax(att, dim=-1).unsqueeze(1)
        for i in range(1, self.m):
            cur_att = torch.bmm(torch.tanh(self.layers[i](enc_hs)), arg.unsqueeze(2)).squeeze()
            cur_att.data.masked_fill_(src_mask.data, -float('inf'))
            cur_att = F.softmax(cur_att, dim=-1).unsqueeze(1)
            att = torch.cat((att, cur_att), 1)
        att = torch.max(att, 1)[0].squeeze()
        sent = torch.bmm(att.unsqueeze(1), enc_hs).squeeze()
        return sent


class GCN(nn.Module):
    def __init__(self, num_layers, in_dim, out_dim):
        self.drop_rate = drop_rate
        super(GCN, self).__init__()
        self.gcn_num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        for i in range(self.gcn_num_layers):
            self.gcn_layers.append(nn.Linear(in_dim, out_dim))
        self.W = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, gcn_input, adj):
        # denom = torch.sum(adj, 2).unsqueeze(2) + 1
        att_scores = torch.bmm(self.W(gcn_input), gcn_input.transpose(1, 2))
        exp_att_scores = torch.exp(att_scores)
        combined_att = adj * exp_att_scores
        denom = torch.sum(combined_att, dim=-1) + 1
        norm_att = combined_att / denom.unsqueeze(2)
        for i in range(self.gcn_num_layers):
            Ax = torch.bmm(norm_att, gcn_input)
            AxW = self.gcn_layers[i](Ax)
            AxW = AxW + self.gcn_layers[i](gcn_input)
            # AxW /= denom
            gAxW = F.relu(AxW)
            gcn_input = self.dropout(gAxW) if i < self.gcn_num_layers - 1 else gAxW
        return gcn_input


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.v = nn.Linear(self.input_dim, 1)

    def forward(self, s_prev, enc_hs, src_mask):
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        # print(attn_weights.shape,src_mask.shape,)
        attn_weights.data.masked_fill_(src_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights


class Encoder(nn.Module):  #编码器
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate
        self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, drop_rate)
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, drop_rate)
        # self.pos_embeddings = nn.Embedding(max_positional_idx, positional_embed_dim, padding_idx=0)
        if enc_type == 'LSTM':
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                                bidirectional=self.is_bidirectional)
        elif enc_type == 'GCN':
            self.reduce_dim = nn.Linear(self.input_dim, 2 * self.hidden_dim)
            self.gcn = GCN(gcn_num_layers, 2* self.hidden_dim, 2 * self.hidden_dim)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                                bidirectional=self.is_bidirectional)
            self.gcn = GCN(gcn_num_layers, 2 * self.hidden_dim, 2 * self.hidden_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)
        # self.mhc = 3
        # self.mha = Multi_Head_Self_Attention(self.mhc, 2 * self.hidden_dim)

    def forward(self, words, chars, pos_seq, adj, is_training=False):
        src_word_embeds = self.word_embeddings(words)
        # pos_embeds = self.dropout(self.pos_embeddings(pos_seq))
        char_embeds = self.char_embeddings(chars)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((src_word_embeds, char_feature), -1)
        if enc_type == 'LSTM':
            outputs, hc = self.lstm(words_input)
        elif enc_type == 'GCN':
            outputs = self.reduce_dim(words_input)
            outputs = self.gcn(outputs, adj)
        else:
            outputs, hc = self.lstm(words_input)
            outputs = self.dropout(outputs)
            outputs = self.gcn(outputs, adj)

        # outputs += pos_embeds
        # outputs = self.mha(outputs, outputs, outputs)
        outputs = self.dropout(outputs)
        return outputs


class Decoder(nn.Module):  #解码器  指针网络+Attention  解码器一次就可以解码出一个三元组的表示
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length

        if att_type == 0:
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(10 * self.input_dim, self.hidden_dim)
        elif att_type == 1:
            # self.w = nn.Linear(9 * self.input_dim, self.input_dim)
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(10 * self.input_dim, self.hidden_dim)
        else:
            # self.w = nn.Linear(9 * self.input_dim, self.input_dim)
            self.attention1 = Attention(input_dim)
            self.attention2 = Attention(input_dim)
            self.lstm = nn.LSTMCell(11 * self.input_dim, self.hidden_dim)

        self.e1_pointer_lstm = nn.LSTM(2 * self.input_dim, self.input_dim, 1, batch_first=True,
                                       bidirectional=True)
        self.e2_pointer_lstm = nn.LSTM(4 * self.input_dim, self.input_dim, 1, batch_first=True,
                                       bidirectional=True)

        self.arg1s_lin = nn.Linear(2 * self.input_dim, 1)
        self.arg1e_lin = nn.Linear(2 * self.input_dim, 1)
        self.arg2s_lin = nn.Linear(2 * self.input_dim, 1)
        self.arg2e_lin = nn.Linear(2 * self.input_dim, 1)
        self.rel_lin = nn.Linear(9 * self.input_dim, len(relnameToIdx))
        self.dropout = nn.Dropout(self.drop_rate)
        self.w = nn.Linear(9 * self.input_dim, self.input_dim)

    def forward(self, y_prev, prev_tuples, h_prev, enc_hs, src_mask, arg1, arg2, arg1_mask, arg2_mask,
                is_training=False):
        src_time_steps = enc_hs.size()[1]
        # print(enc_hs.shape)
        # print(h_prev[0].shape,h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1).shape,self.input_dim)

        if att_type == 0:
            ctx, attn_weights = self.attention(h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        elif att_type == 1:
            reduce_prev_tuples = self.w(prev_tuples)
            ctx, attn_weights = self.attention(reduce_prev_tuples.unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        else:
            # print(src_mask.shape,src_mask,type(src_mask.data),type(src_mask[0][0]),type(src_mask.data[0][0]))
            ctx1, attn_weights1 = self.attention1(h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1),
                                               enc_hs, src_mask)
            reduce_prev_tuples = self.w(prev_tuples)
            ctx2, attn_weights2 = self.attention2(reduce_prev_tuples.unsqueeze(1).repeat(1, src_time_steps, 1),
                                               enc_hs, src_mask)
            ctx = torch.cat((ctx1, ctx2), -1)
            attn_weights = (attn_weights1 + attn_weights2) / 2

        s_cur = torch.cat((prev_tuples, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)

        if use_hadamard:
            enc_hs = enc_hs * attn_weights.unsqueeze(2)
        
        e1_pointer_lstm_input = torch.cat((enc_hs, hidden.unsqueeze(1).repeat(1, src_time_steps, 1)), 2)
        e1_pointer_lstm_out, phc = self.e1_pointer_lstm(e1_pointer_lstm_input)
        e1_pointer_lstm_out = self.dropout(e1_pointer_lstm_out)

        e2_pointer_lstm_input = torch.cat((e1_pointer_lstm_input, e1_pointer_lstm_out), 2)
        e2_pointer_lstm_out, phc = self.e2_pointer_lstm(e2_pointer_lstm_input)
        e2_pointer_lstm_out = self.dropout(e2_pointer_lstm_out)

        arg1s = self.arg1s_lin(e1_pointer_lstm_out).squeeze()
        arg1s.data.masked_fill_(src_mask.data, -float('inf'))

        arg1e = self.arg1e_lin(e1_pointer_lstm_out).squeeze()
        arg1e.data.masked_fill_(src_mask.data, -float('inf'))

        arg2s = self.arg2s_lin(e2_pointer_lstm_out).squeeze()
        arg2s.data.masked_fill_(src_mask.data, -float('inf'))

        arg2e = self.arg2e_lin(e2_pointer_lstm_out).squeeze()
        arg2e.data.masked_fill_(src_mask.data, -float('inf'))

        arg1sweights = F.softmax(arg1s, dim=-1)
        arg1eweights = F.softmax(arg1e, dim=-1)
        ##arg1和arg2都是作为实体的向量表示用于预测关系的
        arg1sv = torch.bmm(arg1eweights.unsqueeze(1), e1_pointer_lstm_out).squeeze()
        arg1ev = torch.bmm(arg1sweights.unsqueeze(1), e1_pointer_lstm_out).squeeze()
        arg1 = self.dropout(torch.cat((arg1sv, arg1ev), -1))

        arg2sweights = F.softmax(arg2s, dim=-1)
        arg2eweights = F.softmax(arg2e, dim=-1)

        arg2sv = torch.bmm(arg2eweights.unsqueeze(1), e2_pointer_lstm_out).squeeze()
        arg2ev = torch.bmm(arg2sweights.unsqueeze(1), e2_pointer_lstm_out).squeeze()
        arg2 = self.dropout(torch.cat((arg2sv, arg2ev), -1))
        
        # enc_hs = self.mha(enc_hs, enc_hs, enc_hs)
        # sent1 = self.mha1(enc_hs, arg1, src_mask)
        # sent2 = self.mha2(enc_hs, arg2, src_mask)

        # if is_training:
        #     # arg1 = self.dropout(multi_head_pooling(mh_hid, arg1_mask, 'sum'))
        #     # arg2 = self.dropout(multi_head_pooling(mh_hid, arg2_mask, 'sum'))
        #
        #     # src_mask = src_mask + arg1_mask.eq(0) + arg2_mask.eq(0)
        #     # src_mask = src_mask.eq(0).eq(0)
        #     sent = self.dropout(multi_head_pooling(mh_hid, src_mask, 'max'))
        # else:
        #     arg1_one_hot = F.gumbel_softmax(arg1s).byte() + F.gumbel_softmax(arg1e).byte()
        #     arg2_one_hot = F.gumbel_softmax(arg2s).byte() + F.gumbel_softmax(arg2e).byte()
        #     # arg1_mask = arg1_one_hot.eq(0)
        #     # arg2_mask = arg2_one_hot.eq(0)
        #
        #     # arg1 = self.dropout(multi_head_pooling(mh_hid, arg1_mask, 'sum'))
        #     # arg2 = self.dropout(multi_head_pooling(mh_hid, arg2_mask, 'sum'))
        #
        #     # src_mask = src_mask + arg1_one_hot + arg2_one_hot
        #     # src_mask = src_mask.eq(0).eq(0)
        #     sent = self.dropout(multi_head_pooling(mh_hid, src_mask, 'max'))

        rel = self.rel_lin(torch.cat((hidden, arg1, arg2), -1))

        if is_training:
            arg1s = F.log_softmax(arg1s, dim=-1)
            arg1e = F.log_softmax(arg1e, dim=-1)
            arg2s = F.log_softmax(arg2s, dim=-1)
            arg2e = F.log_softmax(arg2e, dim=-1)
            rel = F.log_softmax(rel, dim=-1)

            return rel.unsqueeze(1), arg1s.unsqueeze(1), arg1e.unsqueeze(1), arg2s.unsqueeze(1), \
                arg2e.unsqueeze(1), (hidden, cell_state), arg1, arg2
        else:
            arg1s = F.softmax(arg1s, dim=-1)
            arg1e = F.softmax(arg1e, dim=-1)
            arg2s = F.softmax(arg2s, dim=-1)
            arg2e = F.softmax(arg2e, dim=-1)
            rel = F.softmax(rel, dim=-1)
            return rel.unsqueeze(1), arg1s.unsqueeze(1), arg1e.unsqueeze(1), arg2s.unsqueeze(1), arg2e.unsqueeze(1), \
                   (hidden, cell_state), arg1, arg2


class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(enc_inp_size, int(enc_hidden_size/2), 1, True, drop_rate)
        self.decoder = Decoder(dec_inp_size, dec_hidden_size, 1, drop_rate, max_trg_len)
        self.relation_embeddings = nn.Embedding(len(relnameToIdx), word_embed_dim)
        # self.w = nn.Linear(10 * dec_inp_size, dec_inp_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, src_words_seq, src_mask, src_char_seq, pos_seq, trg_words_seq, trg_rel_cnt, adj,
                arg1_mask, arg2_mask, is_training=False):
        if is_training:
            trg_word_embeds = self.dropout(self.relation_embeddings(trg_words_seq))
        batch_len = src_words_seq.size()[0]
        src_time_steps = src_words_seq.size()[1]
        time_steps = trg_rel_cnt
        ##获得的是编码的向量
        enc_hs = self.encoder(src_words_seq, src_char_seq, pos_seq, adj, is_training)
        # print('e',enc_hs.shape)
        ##自此往后都是为解码做准备
        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        dec_hid = (h0, c0)

        dec_inp = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        arg1 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 4 * dec_hidden_size))).cuda()
        arg2 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 4 * dec_hidden_size))).cuda()

        prev_tuples = torch.cat((arg1, arg2, dec_inp), -1)

        if is_training:
            dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, src_mask, arg1, arg2,
                                    arg1_mask[:, 0, :].squeeze(), arg2_mask[:, 0, :].squeeze(), is_training)
        else:
            dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, src_mask, arg1, arg2, None, None,
                                    is_training)
        rel = dec_outs[0]
        arg1s = dec_outs[1]
        arg1e = dec_outs[2]
        arg2s = dec_outs[3]
        arg2e = dec_outs[4]
        dec_hid = dec_outs[5]
        arg1 = dec_outs[6]
        arg2 = dec_outs[7]

        topv, topi = rel[:, :, 1:].topk(1)
        topi = torch.add(topi, 1)

        for t in range(1, time_steps):
            if is_training:
                dec_inp = trg_word_embeds[:, t - 1, :].squeeze()
                prev_tuples = torch.cat((arg1, arg2, dec_inp), -1) + prev_tuples
                dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, src_mask, arg1, arg2,
                                        arg1_mask[:, t, :].squeeze(), arg2_mask[:, t, :].squeeze(), is_training)
            else:
                dec_inp = self.relation_embeddings(topi.squeeze().detach()).squeeze()
                prev_tuples = torch.cat((arg1, arg2, dec_inp), -1) + prev_tuples
                dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, src_mask, arg1, arg2, None, None,
                                        is_training)

            cur_rel = dec_outs[0]
            cur_arg1s = dec_outs[1]
            cur_arg1e = dec_outs[2]
            cur_arg2s = dec_outs[3]
            cur_arg2e = dec_outs[4]
            dec_hid = dec_outs[5]
            arg1 = dec_outs[6]
            arg2 = dec_outs[7]
            ##每解码一个时间步就把解码的向量进行一次拼接
            rel = torch.cat((rel, cur_rel), 1)
            arg1s = torch.cat((arg1s, cur_arg1s), 1)
            arg1e = torch.cat((arg1e, cur_arg1e), 1)
            arg2s = torch.cat((arg2s, cur_arg2s), 1)
            arg2e = torch.cat((arg2e, cur_arg2e), 1)

            topv, topi = cur_rel[:, :, 1:].topk(1)
            topi = torch.add(topi, 1)

        if is_training:
            rel = rel.view(-1, len(relnameToIdx))
            arg1s = arg1s.view(-1, src_time_steps)
            arg1e = arg1e.view(-1, src_time_steps)
            arg2s = arg2s.view(-1, src_time_steps)
            arg2e = arg2e.view(-1, src_time_steps)
        return rel, arg1s, arg1e, arg2s, arg2e


def get_model(model_id):
    if model_id == 1:
        return Seq2SeqModel()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)


def predict(samples, model, model_id):
    pred_batch_size = batch_size
    batch_count = math.ceil(len(samples) / pred_batch_size)  #返回给定数字的最小正整数
   
    move_last_batch = False
    if len(samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    rel = list()
    arg1s = list()
    arg1e = list()
    arg2s = list()
    arg2e = list()
    model.eval()
    set_random_seeds(random_seed)
    start_time = datetime.datetime.now()
    for batch_idx in tqdm(range(0, batch_count)):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_samples_input = get_batch_data(cur_batch, False)

        src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
        positional_seq = torch.from_numpy(cur_samples_input['positional_seq'].astype('long'))
        src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('bool'))
        trg_words_seq = torch.from_numpy(cur_samples_input['decoder_input'].astype('long'))
        src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))
        adj = torch.from_numpy(cur_samples_input['adj'].astype('float32'))

        if torch.cuda.is_available():
            src_words_seq = src_words_seq.cuda()
            src_words_mask = src_words_mask.cuda()
            trg_words_seq = trg_words_seq.cuda()
            src_chars_seq = src_chars_seq.cuda()
            adj = adj.cuda()
            positional_seq = positional_seq.cuda()

        src_words_seq = autograd.Variable(src_words_seq)
        src_words_mask = autograd.Variable(src_words_mask)
        trg_words_seq = autograd.Variable(trg_words_seq)
        src_chars_seq = autograd.Variable(src_chars_seq)
        adj = autograd.Variable(adj)
        positional_seq = autograd.Variable(positional_seq)

        with torch.no_grad():
            if model_id == 1:   #默认就是1
                outputs = model(src_words_seq, src_words_mask, src_chars_seq, positional_seq, trg_words_seq,
                                max_trg_len, adj, None, None, False)

        rel += list(outputs[0].data.cpu().numpy())
        arg1s += list(outputs[1].data.cpu().numpy())
        arg1e += list(outputs[2].data.cpu().numpy())
        arg2s += list(outputs[3].data.cpu().numpy())
        arg2e += list(outputs[4].data.cpu().numpy())
        model.zero_grad()

    end_time = datetime.datetime.now()
    custom_print('Prediction time:', end_time - start_time)
    return rel, arg1s, arg1e, arg2s, arg2e  #rel.shape:10*31  arg.shape:10*91


def train_model(model_id, train_samples, dev_samples, best_model_file):
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if len(train_samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    custom_print(batch_count)
    model = get_model(model_id)   #model_id默认就是1
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    custom_print('Parameters size:', pytorch_total_params)

    custom_print(model)
    if torch.cuda.is_available():
        model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    rel_criterion = nn.NLLLoss(ignore_index=0)
    pointer_criterion = nn.NLLLoss(ignore_index=-1)

    custom_print('weight factor:', wf)
    optimizer = optim.Adam(model.parameters())
    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_epoch_seed = -1
    for epoch_idx in range(0, num_epoch):
        model.train()
        model.zero_grad()
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1

        set_random_seeds(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0

        for batch_idx in tqdm(range(0, batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size)
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(cur_shuffled_train_data)

            cur_batch = cur_shuffled_train_data[batch_start:batch_end]
            cur_samples_input = get_batch_data(cur_batch, True)
            # print(cur_samples_input)
            break
            src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
            positional_seq = torch.from_numpy(cur_samples_input['positional_seq'].astype('long'))
            src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('bool'))
            trg_words_seq = torch.from_numpy(cur_samples_input['decoder_input'].astype('long'))
            src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))
            adj = torch.from_numpy(cur_samples_input['adj'].astype('float32'))

            rel = torch.from_numpy(cur_samples_input['rel'].astype('long'))
            arg1s = torch.from_numpy(cur_samples_input['arg1_start'].astype('long'))
            arg1e = torch.from_numpy(cur_samples_input['arg1_end'].astype('long'))
            arg2s = torch.from_numpy(cur_samples_input['arg2_start'].astype('long'))
            arg2e = torch.from_numpy(cur_samples_input['arg2_end'].astype('long'))

            arg1_mask = torch.from_numpy(cur_samples_input['arg1_mask'].astype('bool'))
            arg2_mask = torch.from_numpy(cur_samples_input['arg2_mask'].astype('bool'))

            if torch.cuda.is_available():
                src_words_seq = src_words_seq.cuda()
                src_words_mask = src_words_mask.cuda()
                trg_words_seq = trg_words_seq.cuda()
                src_chars_seq = src_chars_seq.cuda()
                adj = adj.cuda()
                positional_seq = positional_seq.cuda()

                rel = rel.cuda()
                arg1s = arg1s.cuda()
                arg1e = arg1e.cuda()
                arg2s = arg2s.cuda()
                arg2e = arg2e.cuda()

                arg1_mask = arg1_mask.cuda()
                arg2_mask = arg2_mask.cuda()

            src_words_seq = autograd.Variable(src_words_seq)
            src_words_mask = autograd.Variable(src_words_mask)
            trg_words_seq = autograd.Variable(trg_words_seq)
            src_chars_seq = autograd.Variable(src_chars_seq)
            adj = autograd.Variable(adj)
            positional_seq = autograd.Variable(positional_seq)

            rel = autograd.Variable(rel)
            arg1s = autograd.Variable(arg1s)
            arg1e = autograd.Variable(arg1e)
            arg2s = autograd.Variable(arg2s)
            arg2e = autograd.Variable(arg2e)

            arg1_mask = autograd.Variable(arg1_mask)
            arg2_mask = autograd.Variable(arg2_mask)

            if model_id == 1:
                outputs = model(src_words_seq, src_words_mask, src_chars_seq, positional_seq, trg_words_seq,
                                rel.size()[1], adj, arg1_mask, arg2_mask, True)

            rel = rel.view(-1, 1).squeeze()
            arg1s = arg1s.view(-1, 1).squeeze()
            arg1e = arg1e.view(-1, 1).squeeze()
            arg2s = arg2s.view(-1, 1).squeeze()
            arg2e = arg2e.view(-1, 1).squeeze()

            loss = rel_criterion(outputs[0], rel) + \
                   wf * (pointer_criterion(outputs[1], arg1s) + pointer_criterion(outputs[2], arg1e)) + \
                   wf * (pointer_criterion(outputs[3], arg2s) + pointer_criterion(outputs[4], arg2e))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if (batch_idx + 1) % update_freq == 0:
                optimizer.step()
                model.zero_grad()
            train_loss_val += loss.item()

        train_loss_val /= batch_count
        end_time = datetime.datetime.now()
        custom_print('Training loss:', train_loss_val)
        custom_print('Training time:', end_time - start_time)

        custom_print('\nDev Results\n')
        set_random_seeds(random_seed)
        dev_preds = predict(dev_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(dev_samples, dev_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        dev_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', dev_acc)

        if dev_acc >= best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break

    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Epoch Seed:', best_epoch_seed)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    random_seed = int(sys.argv[2])
    n_gpu = torch.cuda.device_count()
    set_random_seeds(random_seed)

    src_data_folder = sys.argv[3]
    trg_data_folder = sys.argv[4]
    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)
    model_name = 1
    job_mode = sys.argv[5]
    # run = sys.argv[5]
    batch_size = 32
    num_epoch = 30

    max_src_len = 100  #文本的最大句子长度
    max_trg_len = 10   #解码序列的最大长度为10
    embedding_file = os.path.join(src_data_folder, 'w2v.txt')
    update_freq = 1
    wf = 1.0
    att_type = 2

    use_hadamard = False  # bool(int(sys.argv[13]))
    gcn_num_layers = 3
    enc_type = ['LSTM', 'GCN', 'LSTM-GCN'][0]   #默认是用LSTM

    word_embed_dim = 300
    word_min_freq = 2

    char_embed_dim = 50
    char_feature_size = 50
    conv_filter_size = 3
    max_word_len = 10  #词的最大长度为10个字符
    positional_embed_dim = word_embed_dim
    max_positional_idx = 100

    enc_inp_size = word_embed_dim + char_feature_size
    enc_hidden_size = word_embed_dim
    dec_inp_size = enc_hidden_size
    dec_hidden_size = dec_inp_size
    l1_type_embed_dim = 50

    drop_rate = 0.3
    layers = 2
    early_stop_cnt = 5
    sample_cnt = 0
    Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgRels TrgPointers AdjMat")
    rel_file = os.path.join(src_data_folder, 'relations.txt')
    relnameToIdx, relIdxToName = get_relations(rel_file)
    
    # train a model
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')  #记录日志文件
        custom_print(sys.argv)
        custom_print(max_src_len, max_trg_len, drop_rate, layers)
        custom_print(enc_type)
        custom_print('loading data......')
        model_file_name = os.path.join(trg_data_folder, 'model.h5py')

        src_train_file = os.path.join(src_data_folder, 'train.sent')
        trg_train_file = os.path.join(src_data_folder, 'train.pointer')
        adj_train_file = os.path.join(src_data_folder, 'train.dep')  #邻接矩阵文件是预先处理好的
        train_data = read_data(src_train_file, trg_train_file, adj_train_file, 1)

        # print(train_data[0],len(train_data))

        src_dev_file = os.path.join(src_data_folder, 'dev.sent')
        trg_dev_file = os.path.join(src_data_folder, 'dev.pointer')
        adj_dev_file = os.path.join(src_data_folder, 'dev.dep')
        dev_data = read_data(src_dev_file, trg_dev_file, adj_dev_file, 2)

        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))

        custom_print("preparing vocabulary......")
        save_vocab = os.path.join(trg_data_folder, 'vocab.pkl')

        word_vocab, char_vocab, word_embed_matrix = build_vocab(train_data, save_vocab, embedding_file)

        custom_print("Training started......")
        train_model(model_name, train_data, dev_data, model_file_name)
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        custom_print(sys.argv)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, char_vocab = load_vocab(vocab_file_name)

        word_embed_matrix = np.zeros((len(word_vocab), word_embed_dim), dtype=np.float32)
        custom_print('vocab size:', len(word_vocab))

        model_file = os.path.join(trg_data_folder, 'model.h5py')

        best_model = get_model(model_name)
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        if n_gpu > 1:
            best_model = torch.nn.DataParallel(best_model)
        best_model.load_state_dict(torch.load(model_file))

        custom_print('\nTest Results\n')
        src_test_file = os.path.join(src_data_folder, 'test.sent')
        trg_test_file = os.path.join(src_data_folder, 'test.pointer')
        adj_test_file = os.path.join(src_data_folder, 'test.dep')
        test_data = read_data(src_test_file, trg_test_file, adj_test_file, 3)

        reader = open(os.path.join(src_data_folder, 'test.tup'))
        test_gt_lines = reader.readlines()
        reader.close()

        print('Test size:', len(test_data))
        set_random_seeds(random_seed)
        test_preds = predict(test_data, best_model, model_name)
    
        pred_pos, gt_pos, correct_pos = get_F1(test_data, test_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        test_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('P:', round(p, 3))
        custom_print('R:', round(r, 3))
        custom_print('F1:', round(test_acc, 3))
        write_test_res(test_data, test_preds, os.path.join(trg_data_folder, 'test.out'))

        logger.close()


























































