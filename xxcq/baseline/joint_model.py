import torch
import torch.nn as nn
from pytorch_transformers import (WEIGHTS_NAME, BertTokenizer,BertModel, BertPreTrainedModel, BertConfig)
from transformers import BertLayer
import torch.autograd as autograd
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
        attn_weights.data.masked_fill_(src_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights



class Decoder(nn.Module):  #解码器  指针网络+Attention  解码器一次就可以解码出一个三元组的表示
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length,att_type,rel_size):
        super(Decoder, self).__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = layers
        self.drop_rate = int(drop_out_rate)
        self.max_length = int(max_length)

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
        # self.e1_pointer_lstm = nn.GRU(2 * self.input_dim, self.input_dim, 1, batch_first=True,
        #                                bidirectional=True)
        # self.e2_pointer_lstm = nn.GRU(4 * self.input_dim, self.input_dim, 1, batch_first=True,
        #                                bidirectional=True)

        self.arg1s_lin = nn.Linear(2 * self.input_dim, 1)
        self.arg1e_lin = nn.Linear(2 * self.input_dim, 1)
        self.arg2s_lin = nn.Linear(2 * self.input_dim, 1)
        self.arg2e_lin = nn.Linear(2 * self.input_dim, 1)
        self.rel_lin = nn.Linear(9 * self.input_dim, rel_size)
        self.dropout = nn.Dropout(self.drop_rate)
        self.w = nn.Linear(9 * self.input_dim, self.input_dim)

    def forward(self, y_prev, prev_tuples, h_prev, enc_hs, src_mask, arg1, arg2, arg1_mask, arg2_mask,att_type,
                is_training=False):
        # print(enc_hs.shape)
        src_time_steps = enc_hs.size()[1]
        # print(h_prev[0].shape,h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1).shape,self.input_dim)
        if att_type == 0:
            ctx, attn_weights = self.attention(h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        elif att_type == 1:
            reduce_prev_tuples = self.w(prev_tuples)
            ctx, attn_weights = self.attention(reduce_prev_tuples.unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        else:
            # print(src_mask.shape)
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

        # if use_hadamard:
        #     enc_hs = enc_hs * attn_weights.unsqueeze(2)
        
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

def get_span(ner_label,ner_p):
    spanlist=[]
    for i in range(1,len(ner_label)):
        if ner_label[i]==3:
            spanlist.append([[i,i],ner_p[i]])
            for j in range(i+1,len(ner_label)):
                if ner_label[i]==2:
                    spanlist.append([[i,j],(ner_p[i]+ner_p[j])/(j-i+1)])
                else:
                    break
    return spanlist

class BERT_Seq2SeqModel(nn.Module):
    def __init__(self,bertconfig,config):
        super(BERT_Seq2SeqModel, self).__init__()
        # self.encoder = Encoder(enc_inp_size, int(enc_hidden_size/2), 1, True, drop_rate)
        self.encoder = BertModel.from_pretrained( config.model_path, config=bertconfig)
        self.num_labels = bertconfig.num_labels
        self.l2_reg_lambda = bertconfig.l2_reg_lambda
        self.dropout = nn.Dropout(bertconfig.hidden_dropout_prob)
        # self.classifier = nn.Linear(
        #     bertconfig.hidden_size*3, self.num_labels)
        # self.tanh = nn.Tanh()
        vocab_size=config.vocab_size
        self.ner_classifier=nn.Linear(config.enc_hidden_size, vocab_size)

        self.span_layer = BertLayer(config=bertconfig)
        self.w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.gamma = nn.Parameter(torch.ones(1))

        dec_att_type=int(config.dec_att_type)
        self.rel_size=config.rel_size
        self.decoder = Decoder(config.dec_inp_size, config.dec_hidden_size, 1, config.drop_rate, config.max_trg_len,dec_att_type,self.rel_size)
        self.relation_embeddings = nn.Embedding(config.rel_size, config.dec_inp_size)# nn.Embedding(config.rel_size, config.word_embed_dim)
        self.dropout_di = nn.Dropout(config.drop_rate)  #原来模型的dropout函数


    def forward(self, src_words_seq, src_mask, ori_src_words_mask, src_segment, trg_words_seq,  trg_rel_cnt, 
                arg1_mask, arg2_mask,  dec_hidden_size, att_type, input_span_mask, is_training=False):   #src_char_seq, pos_seq,adj,
        if is_training:  ##随机初始化 trg_words_seq是decoder的input
            trg_word_embeds = self.dropout_di(self.relation_embeddings(trg_words_seq))  
            self.encoder.train()
        else:
            self.encoder.eval()
        
        batch_len = src_words_seq.size()[0]
        src_time_steps = src_words_seq.size()[1]  #每个batch的序列长度
        time_steps = trg_rel_cnt
        
        outputs=self.encoder(src_words_seq,attention_mask=src_mask,token_type_ids=src_segment)
        # outputs = self.encoder(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        enc_hs = outputs[0]#outputs[-1]
        # ##获得的是编码的向量
        # enc_hs = self.encoder(src_words_seq, src_char_seq, pos_seq, adj, is_training)
        ner_logits = self.ner_classifier(enc_hs)
        
        # if is_training:
        # span_attention_mask
        extended_span_attention_mask = input_span_mask.unsqueeze(1)
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        # print(enc_hs.shape,extended_span_attention_mask.shape)
        span_sequence_output= self.span_layer(enc_hs, extended_span_attention_mask)
        w = F.softmax(self.w)
        # print(len(span_sequence_output))
        enc_hs = self.gamma * (w[0] * enc_hs + w[1] * span_sequence_output[0])

        # else:
        #     input_span_mask=torch.zeros(batch_len,src_time_steps, src_time_steps, dtype=torch.float)
        #     ner_p,ner_labels=torch.max(F.softmax(ner_logits, dim=-1),-1)
        #     # print(ner_labels.shape,ner_logits.shape,ner_p.shape,ner_labels,ner_p)
        #     for i in range(len(ner_labels)):
        #         spanlist=get_span(ner_labels[i],ner_p[i])
        #         for span in spanlist:
        #             input_span_mask[i][span[0][0]][span[0][1]]=span[1]
        #     input_span_mask=input_span_mask.cuda()
        #     extended_span_attention_mask = input_span_mask.unsqueeze(1)
        #     extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        #     # print(enc_hs.shape,extended_span_attention_mask.shape)
        #     span_sequence_output= self.span_layer(enc_hs, extended_span_attention_mask)
        #     w = F.softmax(self.w)
        #     # print(len(span_sequence_output))
        #     enc_hs = self.gamma * (w[0] * enc_hs + w[1] * span_sequence_output[0])


        ##自此往后都是为解码做准备
        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        dec_hid = (h0, c0)

        dec_inp = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        arg1 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 4 * dec_hidden_size))).cuda()
        arg2 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 4 * dec_hidden_size))).cuda()

        prev_tuples = torch.cat((arg1, arg2, dec_inp), -1)
        # print(enc_hs.shape,dec_inp.shape,prev_tuples.shape)
        if is_training:
            dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, arg1, arg2,
                                    arg1_mask[:, 0, :].squeeze(), arg2_mask[:, 0, :].squeeze(),att_type, is_training)
        else:
            dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, arg1, arg2, None, None,att_type,
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
                # print(dec_inp.shape,arg1.shape,arg2.shape)
                prev_tuples = torch.cat((arg1, arg2, dec_inp), -1) + prev_tuples
                dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, arg1, arg2,
                                        arg1_mask[:, t, :].squeeze(), arg2_mask[:, t, :].squeeze(), att_type, is_training)
            else:
                dec_inp = self.relation_embeddings(topi.squeeze().detach()).squeeze()
                prev_tuples = torch.cat((arg1, arg2, dec_inp), -1) + prev_tuples
                dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, arg1, arg2, None, None,att_type, 
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
            # print(arg1s.shape)
        if is_training:
            rel = rel.view(-1, self.rel_size)
            arg1s = arg1s.view(-1, src_time_steps)
            arg1e = arg1e.view(-1, src_time_steps)
            arg2s = arg2s.view(-1, src_time_steps)
            arg2e = arg2e.view(-1, src_time_steps)
        return rel, arg1s, arg1e, arg2s, arg2e, ner_logits

