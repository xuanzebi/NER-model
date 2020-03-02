import torch
import torch.nn as nn
from model.lstm.crf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss, MSELoss
import codecs
import json
import gc
import numpy as np
from transformers.modeling_bert import BertPreTrainedModel,BertModel,BertConfig

import sys
package_dir_b = "/opt/hyp/project/ELMoForManyLangs"
sys.path.insert(0, package_dir_b)
from elmoformanylangs import Embedder

class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(Highway, self).__init__()

        self._layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self._layers:
            # Bias the highway layer to just carry its input forward.
            # Set the bias on B(x) to be positive, then g will be biased to be high
            # The bias on B(x) is the second half of the bias vector in each linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self._layers:
            linear_part = current_inputs
            projected_inputs = layer(current_inputs)

            nonlinear_part, gate = projected_inputs.chunk(2, dim=-1)
            nonlinear_part = torch.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_inputs = gate * linear_part + (1 - gate) * nonlinear_part
        return current_inputs


class WordRep(nn.Module):
    """
    词向量：glove/字向量/elmo/bert/flair
    """

    def __init__(self, args, pretrain_word_embedding):
        super(WordRep, self).__init__()
        self.word_emb_dim = args.word_emb_dim
        self.char_emb_dim = args.char_emb_dim
        self.use_char = args.use_char
        self.use_pre = args.use_pre
        self.freeze = args.freeze
        self.drop = nn.Dropout(args.dropout)

        if self.use_pre:
            if self.freeze:
                self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_word_embedding),
                                                                   freeze=True).float()
            else:
                self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_word_embedding),
                                                                   freeze=False).float()
        else:
            self.word_embedding = nn.Embedding(args.vocab_size, 300)

        if self.use_char:
            pass

    def forward(self, word_input):
        word_embs = self.word_embedding(word_input)
        word_represent = self.drop(word_embs)
        return word_represent

def get_elmo_embedding(word_input,input_mask,guid,embeddings):
    bert_emb = torch.zeros(len(word_input),200,1024,dtype=torch.float)

    for i, gu_id in enumerate(guid):
        tmp = torch.from_numpy(embeddings[gu_id])
        if tmp.size(0) < 200:
            a = torch.zeros(200-tmp.size(0),1024,dtype=torch.float)
            tmp = torch.cat((tmp,a),0)

        bert_emb[i] = tmp

    return bert_emb
    

def get_bert_embedding(word_input,input_mask,guids,embeddings):
    """每一个line是一条文本数据，考虑将"""
    bert_emb = torch.zeros(len(word_input),200,768,dtype=torch.float)
    for i,te in enumerate(guids):
        text_len = sum(input_mask[i] == 1)
        tmp = torch.from_numpy(embeddings[te])

        if len(embeddings[te]) < text_len:
            a = torch.randn(text_len-len(embeddings[te]),768,dtype=torch.float)
            tmp = torch.cat((tmp,a),0)

        if tmp.size(0) < 200:
            a = torch.zeros(200-tmp.size(0),768,dtype=torch.float)
            tmp = torch.cat((tmp,a),0)

        bert_emb[i] = tmp

    return bert_emb

# TODO  尝试使用官方代码，测试效果
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class Bilstmcrf(nn.Module):
    """
    bilstm-crf模型
    """

    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Bilstmcrf, self).__init__()
        self.use_crf = args.use_crf
        self.use_char = args.use_char
        self.gpu = args.gpu
        self.use_pre = args.use_pre
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.rnn_type = args.rnn_type
        self.max_seq_length = args.max_seq_length
        self.use_highway = args.use_highway
        self.dropoutlstm = nn.Dropout(args.dropoutlstm)
        self.wordrep = WordRep(args, pretrain_word_embedding)
        self.use_elmo = args.use_elmo
        self.use_bert = args.use_bert
        self.args = args
        self.word_emb_dim = args.word_emb_dim
        
        if self.use_bert:
            if not self.args.test:
                self.train_bert_embeddings = self._load_elmo_bert_embedding('/opt/hyp/NER/NER-model/data/bert/embedding/cyber_bert_train_embedding.txt','bert')
                self.dev_bert_embeddings = self._load_elmo_bert_embedding('/opt/hyp/NER/NER-model/data/bert/embedding/cyber_bert_dev_embedding.txt','bert')
            if self.args.test:
                self.test_bert_embeddings = self._load_elmo_bert_embedding('/opt/hyp/NER/NER-model/data/bert/embedding/cyber_bert_test_embedding.txt','bert')
        if self.use_elmo:
            if not self.args.test:
                self.train_elmo_embeddings = self._load_elmo_bert_embedding('/opt/hyp/NER/NER-model/data/elmo/cyber_elmo_train.txt','elmo')
                self.dev_elmo_embeddings = self._load_elmo_bert_embedding('/opt/hyp/NER/NER-model/data/elmo/cyber_elmo_dev.txt','elmo')
            if self.args.test:
                self.test_elmo_embeddings = self._load_elmo_bert_embedding('/opt/hyp/NER/NER-model/data/elmo/cyber_elmo_test.txt','elmo')

        gc.collect()

        if self.use_elmo:
            self.word_emb_dim = args.word_emb_dim + 1024

        self.lstm = nn.LSTM(self.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(self.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                          bidirectional=True)

        self.label_size = label_size
        if self.use_crf:
            self.crf = CRF(self.label_size, self.gpu)
            self.label_size += 2
        if self.use_highway:
            self.highway = Highway(args.rnn_hidden_dim * 2, 1)

        self.hidden2tag = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)

    def _load_elmo_bert_embedding(self,path,mode):
        embeddings = []
        if mode == 'elmo':
            emb_size = 1024
        elif mode == 'bert':
            emb_size = 768

        with open(path,'r',encoding='utf-8') as fr:
            for line in fr:
                if line:
                    a = np.asarray(line.strip().split(' '), dtype='float32').reshape(-1, emb_size)
                    embeddings.append(a)
        return embeddings

    # pack_padded  pad_packed_sequence
    def forward_pack(self, word_input, input_mask, labels):
        # word_input input_mask   FloatTensor
        word_input = self.wordrep(word_input)

        input_mask.requires_grad = False
        word_input = word_input * (input_mask.unsqueeze(-1).float())
        batch_size = word_input.size(0)

        total_length = word_input.size(1)
        ttt = input_mask.ge(1)
        word_seq_lengths = [int(torch.sum(i).cpu().numpy()) for i in ttt]

        if self.rnn_type == 'LSTM':
            packed_words = pack_padded_sequence(word_input, word_seq_lengths, True, enforce_sorted=False)
            lstm_out, hidden = self.lstm(packed_words)
            output, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)
        elif self.rnn_type == 'GRU':
            packed_words = pack_padded_sequence(word_input, word_seq_lengths, True, enforce_sorted=False)
            lstm_out, hidden = self.gru(packed_words)
            output, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)

        if self.use_highway:
            output = self.highway(output)

        output = self.dropoutlstm(output)
        output = self.hidden2tag(output)
        maskk = input_mask.ge(1)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output, input_mask)
            return total_loss / batch_size, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            active_loss = input_mask.view(-1) == 1
            active_logits = output.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss, output

    def forward(self, word_input, input_mask, labels,guids=None,mode=None):
        # word_input input_mask   FloatTensor
        word_input_id = self.wordrep(word_input)

        if self.use_elmo:
            if mode == 'train':
                elmo_embedding = get_elmo_embedding(word_input,input_mask,guids,self.train_elmo_embeddings)
            elif mode == 'test':
                elmo_embedding = get_elmo_embedding(word_input,input_mask,guids,self.test_elmo_embeddings)
            elif mode == 'dev':
                elmo_embedding = get_elmo_embedding(word_input,input_mask,guids,self.dev_elmo_embeddings)
            elmo_embedding = elmo_embedding.to('cuda')
            word_input_id = torch.cat((word_input_id,elmo_embedding),-1)
        if self.use_bert:
            if mode == 'train':
                bert_embedding = get_bert_embedding(word_input,input_mask,guids,self.train_bert_embeddings)
            elif mode == 'test':
                bert_embedding = get_bert_embedding(word_input,input_mask,guids,self.test_bert_embeddings)
            elif mode == 'dev':
                bert_embedding = get_bert_embedding(word_input,input_mask,guids,self.dev_bert_embeddings)
            bert_embedding = bert_embedding.to('cuda')
            word_input_id = torch.cat((word_input_id,bert_embedding),1)

        input_mask.requires_grad = False
        word_input_id = word_input_id * (input_mask.unsqueeze(-1).float())

        batch_size = word_input_id.size(0)
        if self.rnn_type == 'LSTM':
            output, _ = self.lstm(word_input_id)
        elif self.rnn_type == 'GRU':
            output, _ = self.gru(word_input_id)

        if self.use_highway:
            output = self.highway(output)

        output = self.dropoutlstm(output)
        output = self.hidden2tag(output)
        maskk = input_mask.ge(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output, input_mask)
            return total_loss / batch_size, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)

            active_loss = input_mask.view(-1) == 1
            active_logits = output.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            return loss, output
