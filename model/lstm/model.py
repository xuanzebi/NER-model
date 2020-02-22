import torch
import torch.nn as nn
from model.lstm.crf import CRF

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


class Bilstm_CRF_MTL(nn.Module):
    """
    bilstm-crf模型 + 多任务学习(预测实体的token<尝试过CRF与不过>，以及使用多个数据来多任务学习共享参数)
    # TODO 将 token_loss过 crf   loss设置不同权重
    """

    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Bilstm_CRF_MTL, self).__init__()
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

        self.lstm = nn.LSTM(args.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(args.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                          bidirectional=True)

        self.label_size = label_size
        if self.use_crf:
            self.crf = CRF(self.label_size, self.gpu)
            self.label_size += 2
            # self.token_crf = CRF(2,self.gpu,-1)

        if self.use_highway:
            self.highway = Highway(args.rnn_hidden_dim * 2, 1)

        self.hidden2tag = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)

        self.hidden2token = nn.Linear(args.rnn_hidden_dim * 2,2)

    def forward(self, word_input, input_mask, labels,labels_token):
        # word_input input_mask   FloatTensor
        word_input = self.wordrep(word_input)
        input_mask.requires_grad = False
        word_input = word_input * (input_mask.unsqueeze(-1).float())
        batch_size = word_input.size(0)

        if self.rnn_type == 'LSTM':
            output, _ = self.lstm(word_input)
        elif self.rnn_type == 'GRU':
            output, _ = self.gru(word_input)

        if self.use_highway:
            output = self.highway(output)

        output2 = self.dropoutlstm(output)
        output2 = self.hidden2tag(output2)

        token_output = self.hidden2token(output)


        loss_token = nn.CrossEntropyLoss(ignore_index=0)
        active_loss = input_mask.view(-1) == 1

        active_logits = token_output.view(-1, 2)[active_loss]
        active_labels = labels_token.view(-1)[active_loss]
        token_loss = loss_token(active_logits, active_labels)

        maskk = input_mask.ge(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output2, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output2, input_mask)
            
            # token_loss = self.token_crf.neg_log_likelihood_loss(token_output,maskk,labels_token)
            ans_loss = total_loss / batch_size * 0.7 + token_loss * 0.3
            return ans_loss, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            active_loss = input_mask.view(-1) == 1
            active_logits = output2.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss + token_loss, output2


class Bilstm_ST_END(nn.Module):
    """
    bilstm-crf模型
    """

    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Bilstm_ST_END, self).__init__()
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

        self.lstm = nn.LSTM(args.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(args.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                          bidirectional=True)

        self.label_size = label_size

        self.hidden2start = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)
        self.hidden2end = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)


    def forward(self, word_input, input_mask, start_id,end_id):
        # word_input input_mask   FloatTensor
        word_input = self.wordrep(word_input)

        input_mask.requires_grad = False
        word_input = word_input * (input_mask.unsqueeze(-1).float())

        batch_size = word_input.size(0)
        if self.rnn_type == 'LSTM':
            output, _ = self.lstm(word_input)
        elif self.rnn_type == 'GRU':
            output, _ = self.gru(word_input)

        if self.use_highway:
            output = self.highway(output)

        output = self.dropoutlstm(output)

        start_output = self.hidden2start(output)
        end_output = self.hidden2end(output)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        active_loss = input_mask.view(-1) == 1
        start_logits = start_output.view(-1, self.label_size)[active_loss]
        start_labels = start_id.view(-1)[active_loss]
        start_loss = loss_fct(start_logits, start_labels)

        end_logits = end_output.view(-1, self.label_size)[active_loss]
        end_labels = end_id.view(-1)[active_loss]
        end_loss = loss_fct(end_logits, end_labels)
        
        return start_loss + end_loss, start_output, end_output