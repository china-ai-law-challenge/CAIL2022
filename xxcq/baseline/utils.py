import sys
import os
import numpy as np
import random

from configparser import ConfigParser
from collections import OrderedDict
import pickle
import datetime
import json
from tqdm import tqdm
from recordclass import recordclass  #提供tuple结构，占用内存少
import math
from joint_model import BERT_Seq2SeqModel
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
n_gpu = torch.cuda.device_count()

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
    # nameToIdx['None'] = 1
    # idxToName[1] = 'None'
    idx = 1
    for line in lines:
        nameToIdx[line.strip()] = idx
        idxToName[idx] = line.strip()
        idx += 1
    return nameToIdx, idxToName

rel_file = os.path.join('./', 'relations.txt')
# rel_file = os.path.join('./NYT29', 'relations.txt')
relnameToIdx, relIdxToName = get_relations(rel_file)
# print(relIdxToName)
def get_data(src_lines, trg_lines, datatype):#获取数据、将json格式的数据处理成需要的格式 存进samples数组中
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

        # adj_data = json.loads(adj_lines[i])
        # adj_mat = get_adj_mat(len(src_words), adj_data['adj_mat'])

        for part in parts:
            elements = part.strip().split()
            trg_rels.append(relnameToIdx[elements[4]])
            trg_pointers.append((int(elements[0]), int(elements[1]), int(elements[2]), int(elements[3])))

        if datatype == 1 and (len(src_words) > max_src_len or len(trg_rels) > max_trg_len):
            continue

        sample = Sample(Id=uid, SrcLen=len(src_words), SrcWords=src_words, TrgLen=len(trg_rels), TrgRels=trg_rels,
                        TrgPointers=trg_pointers) #, AdjMat=adj_mat
        samples.append(sample)
        uid += 1
    return samples


def read_data(src_file, trg_file, datatype):  #从处理好的文档中读取数据、src是文本文件、trg是指针文件也就是结果对应的序列、adj是邻接矩阵、datatype表示训练开发和测试、训练集中数据需要随机打乱
    reader = open(src_file)
    src_lines = reader.readlines()
    reader.close()

    reader = open(trg_file)
    trg_lines = reader.readlines()
    reader.close()

    # reader = open(adj_file)
    # adj_lines = reader.readlines()
    # reader.close()

    # l = 1000
    # src_lines = src_lines[0:min(l, len(src_lines))]
    # trg_lines = trg_lines[0:min(l, len(trg_lines))]
    # adj_lines = adj_lines[0:min(l, len(adj_lines))]

    data = get_data(src_lines, trg_lines, datatype)
    return data

def get_model(model_id,bertconfig,config):
    if model_id == 1:
        return BERT_Seq2SeqModel(bertconfig,config)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)


def shuffle_data(data,config):  #按batch随机打乱数据
    # custom_print(len(data))  #print变量、写日志文件
    batch_size=config.per_gpu_train_batch_size
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

def get_padded_pointers(pointers, pidx, max_len):
    idx_list = []
    for p in pointers:
        idx_list.append(p[pidx])
    pad_len = max_len + 1 - len(pointers)
    for i in range(0, pad_len):
        idx_list.append(-1)
    return idx_list

def get_padded_mask(cur_len, max_len):
    mask_seq = list()
    for i in range(0, cur_len):
        mask_seq.append(0)
    pad_len = max_len - cur_len
    for i in range(0, pad_len):
        mask_seq.append(1)
    return mask_seq


def get_padded_relations(rels, max_len):
    rel_list = []
    for r in rels:
        rel_list.append(r)
    rel_list.append(relnameToIdx['NA'])
    pad_len = max_len + 1 - len(rel_list)
    for i in range(0, pad_len):
        rel_list.append(relnameToIdx['<PAD>'])
    return rel_list


def get_relation_index_seq(rel_ids, max_len):
    seq = list()
    # seq.append(relnameToIdx['<SOS>'])
    for r in rel_ids:
        seq.append(r)
    seq.append(relnameToIdx['NA'])
    pad_len = max_len + 1 - len(seq)
    for i in range(0, pad_len):
        seq.append(relnameToIdx['<PAD>'])
    return seq


def get_entity_masks(pointers, src_max, trg_max):
    arg1_masks = []
    arg2_masks = []
    for p in pointers:
        arg1_mask = [1 for i in range(src_max)]
        arg1_mask[p[0]] = 0
        arg1_mask[p[1]] = 0

        arg2_mask = [1 for i in range(src_max)]
        # print(src_max,p[2])
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

def get_span_pos(pointers,max_seq_length):
    span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)
    for p in pointers:
        for i in range(p[0]+1,p[1]+2):
            for j in range(i+1,p[1]+2):
                span_pos[i][j]=1
        for i in range(p[2]+1,p[3]+2):
            for j in range(i+1,p[3]+2):
                span_pos[i][j]=1
        # span_pos[p[2]+1][p[3]+1]=1
        # span_pos[p[2]+1][p[3]+1]=1
    return span_pos

def get_span_pos1(spanTag,max_seq_length):
    span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)
    for i in range(0,len(spanTag),2):
        span_pos[int(spanTag[i])+1][int(spanTag[i+1])+1]=1
    return span_pos


def get_ner_feature(nerlist):
    res=[]
    for x in nerlist:
        if x==0 or x==1:
            res.append(0)
        else:
            res.append(1)
    return res

def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_src_max_len, batch_trg_max_len = get_max_len(cur_samples)
    batch_trg_max_len += 1
    src_words_list = list()
    src_words_mask_list = list()
    ori_src_words_mask_list = list()
    src_seg_list=list()
    # src_char_seq = list()
    decoder_input_list = list()
    ner_list = []
    # adj_lst = []
    # positional_index_list = []

    rel_seq = list()
    arg1_start_seq = list()
    arg1_end_seq = list()
    arg2_start_seq = list()
    arg2_end_seq = list()
    arg1_mask_seq = []
    arg2_mask_seq = []
    ner_feature_list = []

    for sample in cur_samples:
        # print(len(sample.input_ids),sample.SrcLen,len(sample.input_mask))
        src_words_list.append(sample.input_ids)
        src_words_mask_list.append(sample.input_mask)
        src_seg_list.append(sample.segment_ids)
        ner_list.append(sample.nerTag)
        # print(ner_list)
        # src_words_list.append(get_words_index_seq(sample.SrcWords, batch_src_max_len))
        ori_src_words_mask_list.append(get_padded_mask(sample.SrcLen, len(sample.input_ids)))#batch_src_max_len))
        # src_char_seq.append(get_char_seq(sample.SrcWords, batch_src_max_len))
        # cur_masked_adj = np.zeros((batch_src_max_len, batch_src_max_len), dtype=np.float32)
        # cur_masked_adj[:len(sample.SrcWords), :len(sample.SrcWords)] = sample.AdjMat
        # adj_lst.append(cur_masked_adj)
        # positional_index_list.append(get_positional_index(len(sample.SrcWords), batch_src_max_len))
        # print(sample.spanTag)
        # ner_feature_list.append(get_span_pos(sample.TrgPointers,len(sample.input_ids)))
        ner_feature_list.append(get_span_pos1(sample.spanTag,len(sample.input_ids)))
        if is_training:
            arg1_start_seq.append(get_padded_pointers(sample.TrgPointers, 0, batch_trg_max_len))
            arg1_end_seq.append(get_padded_pointers(sample.TrgPointers, 1, batch_trg_max_len))
            arg2_start_seq.append(get_padded_pointers(sample.TrgPointers, 2, batch_trg_max_len))
            arg2_end_seq.append(get_padded_pointers(sample.TrgPointers, 3, batch_trg_max_len))
            rel_seq.append(get_padded_relations(sample.TrgRels, batch_trg_max_len))
            decoder_input_list.append(get_relation_index_seq(sample.TrgRels, batch_trg_max_len))
            # print(sample.SrcWords,batch_src_max_len,len(sample.SrcWords))
            arg1_mask, arg2_mask = get_entity_masks(sample.TrgPointers, batch_src_max_len, batch_trg_max_len)
            arg1_mask_seq.append(arg1_mask)
            arg2_mask_seq.append(arg2_mask)
            
            # print(ner_feature_list)
        else:
            decoder_input_list.append(get_relation_index_seq([], 1))

    return {'src_words': np.array(src_words_list, dtype=np.float32),
            'src_words_mask': np.array(src_words_mask_list),
            'ori_src_words_mask':np.array(ori_src_words_mask_list),
            'nerTag':np.array(ner_list),
            'src_segment':np.array(src_seg_list),
            'decoder_input': np.array(decoder_input_list),
            'rel': np.array(rel_seq),
            'arg1_start':np.array(arg1_start_seq),
            'arg1_end': np.array(arg1_end_seq),
            'arg2_start': np.array(arg2_start_seq),
            'arg2_end': np.array(arg2_end_seq),
            'arg1_mask': np.array(arg1_mask_seq),
            'arg2_mask': np.array(arg2_mask_seq),
            'ner_feature':np.array(ner_feature_list)}
            # 'positional_seq': np.array(positional_index_list),
            # 'src_chars': np.array(src_char_seq),
            # 'adj': np.array(adj_lst),


def is_full_match(triplet, triplets):  #判断三元组triplet是否在triplets三元组列表中
    for t in triplets:
        if t[0] == triplet[0] and t[1] == triplet[1] and t[2] == triplet[2]:
            return True
    return False


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


def get_gt_triples(src_words, rels, pointers):  #获得真实的三元组
    triples = []
    i = 0
    for r in rels:
        # print(type(src_words[0]))
        arg1 = ''.join(src_words[pointers[i][0]:pointers[i][1] + 1])
        arg2 = ''.join(src_words[pointers[i][2]:pointers[i][3] + 1])
        triplet = (arg1.strip(), arg2.strip(), relIdxToName[r])
        if not is_full_match(triplet, triples):
            triples.append(triplet)
        i += 1
    return triples


def get_pred_triples(rel, arg1s, arg1e, arg2s, arg2e, src_words):  #获得当前语句中预测的三元组 返回无重复的三元组列表和有重复的三元列表
    triples = []
    all_triples = []
    
    ##len(rel)=10  表示解码序列的最大长度为10
    for i in range(0, len(rel)):
        # print(rel.shape,rel)
        r = np.argmax(rel[i][1:]) + 1
        if r == relnameToIdx['NA']:
            break
        s1, e1, s2, e2 = get_answer_pointers(arg1s[i], arg1e[i], arg2s[i], arg2e[i], len(src_words))
        arg1 = ''.join(src_words[s1: e1 + 1 ])  #+ 1
        arg2 = ''.join(src_words[s2: e2 + 1])#+ 1
        arg1 = arg1.strip()
        arg2 = arg2.strip()
        # print(s1,e1,s2,e2,arg1,arg2)
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
        # print('rel arg1s',preds[0][i].shape,preds[1][i].shape,preds[0][i], preds[1][i], preds[2][i], preds[3][i],
        #                                                   preds[4][i])
        # print(len(preds[1][i]),preds[1][i])
        pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                          preds[4][i], data[i].SrcWords)
        total_pred_pos += len(all_pred_triples)
        gt_pos += len(gt_triples)
        pred_pos += len(pred_triples)
        # print(gt_triples)
        # print(pred_triples)
        for gt_triple in gt_triples:
            if is_full_match(gt_triple, pred_triples):
                correct_pos += 1
            else:
                print(gt_triple,pred_triples)
    # print('t:',total_pred_pos)
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


class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()
        raw_config.read(config_file)
        self.cast_values(raw_config)
        

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                    
                setattr(self, key, val)

