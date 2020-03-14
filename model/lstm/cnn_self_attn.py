import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lstm.crf import CRF
from model.lstm.helper import Highway,WordRep
from model.helper.adversarial_model import adversarial_train_loss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss, MSELoss
import codecs
import numpy as np

# n_position 为句子划分成字符或者词的长度，d_hid为词向量的维度。
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  偶数正弦
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  奇数余弦

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.tensor(sinusoid_table, dtype=torch.float)  # n_position × d_hid  得到每一个词的位置向量

class CNN_BASE(nn.Module):
    def __init__(self, args, pretrain_word_embedding, label_size,mode):
        super(CNN_BASE, self).__init__()
        self.use_crf = args.use_crf
        self.gpu = args.gpu
        self.use_pre = args.use_pre
        self.max_seq_length = args.max_seq_length
        self.pos_emb = args.pos_emb
        self.use_multi_token_mtl = args.use_multi_token_mtl
        self.dropout = nn.Dropout(0.1)
        self.mode = mode
        self.word_emb_dim = args.word_emb_dim
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.word_emb_dim) 
        self.wordrep = WordRep(args, pretrain_word_embedding)
        self.label_size = label_size
        self.cnn_dim = args.cnn_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=args.word_emb_dim, out_channels=self.cnn_dim, kernel_size=3,
                      padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
                      ),
            nn.ReLU(),
        )

        self.cnn_glu = nn.Sequential(
            nn.Conv1d(in_channels=args.word_emb_dim, out_channels=self.cnn_dim, kernel_size=3,
                      padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
                      ),
            nn.GLU(dim=1),
        )

        self.inter_media = nn.Sequential(nn.Linear(self.cnn_dim,100),
                                nn.ReLU())

        ###  池化
        self.max_pool = nn.MaxPool1d(args.max_seq_length) # 200是句子单词的长度
        self.max_fea_pool = nn.MaxPool1d(self.cnn_dim) # 200是句子单词的长度

        if self.use_crf:
            self.crf = CRF(self.label_size, self.gpu)
            self.label_size += 2
            # self.token_crf = CRF(2,self.gpu,-1)

        if self.mode == 'GCNN':
            self.hidden2tag = nn.Linear(100, self.label_size)
        elif self.mode == 'GLDR':
            self.hidden2tag = nn.Linear(100, self.label_size)
        else:
            self.hidden2tag = nn.Linear(self.cnn_dim, self.label_size)

    def forward(self, word_input, input_mask, labels):
        # word_input input_mask   FloatTensor
        word_input = self.wordrep(word_input)

        if self.pos_emb == 1:
            pos_emb = self.get_position_embedding(learn_pos=True,word_input=word_input)
            pos_emb = self.position_embeddings(pos_emb)
            word_input = word_input + pos_emb
        elif self.pos_emb == 2:
            pos_emb = self.get_position_embedding(learn_pos=False,word_input=word_input).to('cuda')
            word_input = word_input + pos_emb

        input_mask.requires_grad = False
        word_input = word_input * (input_mask.unsqueeze(-1).float())
        batch_size = word_input.size(0)

        word_input = word_input.transpose(2, 1).contiguous()

        if self.mode == 'GCNN':
            word_input_id = self.gated_resnet_torch(word_input,3,self.cnn_dim)
            word_input_id = self.gated_resnet_torch(word_input_id,3,self.cnn_dim)
            word_input_id = self.gated_resnet_torch(word_input_id,3,self.cnn_dim)
            word_input_id = self.gated_resnet_torch(word_input_id,3,self.cnn_dim)
            word_input_id = self.gated_resnet_torch(word_input_id,3,self.cnn_dim)
            word_input_id = self.gated_resnet_torch(word_input_id,3,self.cnn_dim)
            word_input_id = word_input_id.transpose(2, 1).contiguous()
            cnn_output = self.inter_media(word_input_id)
        elif self.mode == 'CNN':
            cnn_output = self.cnn(word_input).transpose(2, 1).contiguous()
        elif self.mode == 'GLDR':
            word_input_id = self.cnn_glu(word_input)
            word_input_id = self.gated_linear_d_r(word_input_id,dilation=1,kernel_size=3,dim=word_input_id.shape[1])
            word_input_id = self.gated_linear_d_r(word_input_id,dilation=2,kernel_size=3,dim=word_input_id.shape[1])
            word_input_id = self.gated_linear_d_r(word_input_id,dilation=4,kernel_size=3,dim=word_input_id.shape[1])
            word_input_id = self.gated_linear_d_r(word_input_id,dilation=8,kernel_size=3,dim=word_input_id.shape[1])
            word_input_id = self.gated_linear_d_r(word_input_id,dilation=16,kernel_size=3,dim=word_input_id.shape[1])
            word_input_id = self.gated_linear_d_r(word_input_id,dilation=1,kernel_size=3,dim=word_input_id.shape[1])
            word_input_id = self.gated_linear_d_r(word_input_id,dilation=1,kernel_size=3,dim=word_input_id.shape[1])
            cnn_output = self.gated_linear_d_r(word_input,dilation=1,kernel_size=3,dim=word_input.shape[1])
            cnn_output = cnn_output.transpose(2, 1).contiguous()

        output = self.dropout(cnn_output)
        output2 = self.hidden2tag(output)
        maskk = input_mask.ge(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output2, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output2, input_mask)
            total_loss = total_loss / batch_size
            return total_loss, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            active_loss = input_mask.view(-1) == 1
            active_logits = output2.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss, output2

    def get_position_embedding(self,learn_pos,word_input):
        # 位置向量
        if learn_pos:
            # 可学习的位置向量
            position_ids = torch.arange(self.max_seq_length, dtype=torch.long, device='cuda')
            position_ids = position_ids.unsqueeze(0).expand((word_input.shape[:2]))
        else:
            position_ids = get_sinusoid_encoding_table(self.max_seq_length, self.word_emb_dim).unsqueeze(0).expand(word_input.size()) # 3 为 batch_size 5: sel_len 6：hidden_dim

        return position_ids
    
    def gated_resnet_torch(self,x, kernel_size, dim):
        # # 门卷积 + 残差  GCNN
        # conv1 = nn.Conv1d(x.shape[1], dim, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        # conv2 = nn.Conv1d(x.shape[1], dim, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        # x1 = conv1(x)
        # x2 = conv2(x)
        # x2 = torch.sigmoid(x2)
        # print(x1 + x2)
        # 另一种
        conv = nn.Conv1d(x.shape[1], dim * 2, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2)).to('cuda')
        conv_x = conv(x)
        x = x + F.glu(conv_x, 1)
        # batchnorm = nn.BatchNorm1d(self.cnn_dim).to('cuda') batchnorm(x) 

        return x
    
    def gated_linear_d_r(self,x,dilation,kernel_size,dim):
        drop1 = nn.Dropout(0.2).to('cuda')
        drop2 = nn.Dropout(0.2).to('cuda')
        y = drop1(x)
        conv1 = nn.Conv1d(x.shape[1], dim * 2, kernel_size=kernel_size,dilation=dilation, padding=int((kernel_size - 1) * dilation / 2)).to('cuda')
        act1 = nn.GLU(dim=1).to('cuda')
        y = act1(conv1(y))
        y = drop2(y)
        conv2= nn.Conv1d(x.shape[1], dim * 2, kernel_size=kernel_size, dilation=dilation,padding=int((kernel_size - 1) * dilation/ 2)).to('cuda')
        act2 = nn.GLU(dim=1).to('cuda')
        y = act2(conv2(y))

        return x + y


    def get_output(self,word_input,input_mask,pool_mode=None):
        """
                    word_input: batch_size * seq_len * dim
            RETURN: cnn_output: batch_size * seq_len(padding kernel_size) * out_channels 
        """
        # word_input input_mask   FloatTensor
        # word_input = self.wordrep(word_input)
        if self.pos_emb == 1:
            pos_emb = self.get_position_embedding(learn_pos=True,word_input=word_input)
            pos_emb = self.position_embeddings(pos_emb)
            word_input = word_input + pos_emb
        elif self.pos_emb == 2:
            pos_emb = self.get_position_embedding(learn_pos=False,word_input=word_input).to('cuda')
            word_input = word_input + pos_emb

        batch_size = word_input.size(0)

        word_input = word_input.transpose(2, 1).contiguous()

        if self.mode == 'GCNN':
            word_input = self.gated_resnet_torch(word_input,3,self.cnn_dim)
            word_input = self.gated_resnet_torch(word_input,3,self.cnn_dim)
            word_input = self.gated_resnet_torch(word_input,3,self.cnn_dim)
            word_input = self.gated_resnet_torch(word_input,3,self.cnn_dim)
            word_input = self.gated_resnet_torch(word_input,3,self.cnn_dim)
            word_input = self.gated_resnet_torch(word_input,3,self.cnn_dim)
            cnn_output = word_input.transpose(2, 1).contiguous()
            # cnn_output = self.inter_media(cnn_output)
        elif self.mode == 'CNN':
            cnn_output = self.cnn(word_input).transpose(2, 1).contiguous()
        elif self.mode == 'CNNPOOLING':
            cnn_output = self.cnn(word_input)
            if pool_mode == 'dim': # 返回 b * 1 * cnn_dim  
                cnn_output = self.max_pool(cnn_output)
                cnn_output = cnn_output.transpose(2, 1).contiguous()
            elif pool_mode == 'seq': # 返回 b * seq_len * 1
                cnn_output = self.max_fea_pool(cnn_output.transpose(2, 1).contiguous())
        
        # output2 = self.dropout(cnn_output)
        return cnn_output



class Self_Attn(nn.Module):
    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Self_Attn, self).__init__()
    
    def forward(self, word_input, input_mask, labels,labels_token=None):
        pass

    def get_output(self,word_input,input_mask):
        pass

class Bilstmcrf_cnn(nn.Module):
    """
    bilstm-crf模型
    """

    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Bilstmcrf_cnn, self).__init__()
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
        self.args = args
        self.multi_token_loss_alpha = args.multi_token_loss_alpha
        self.word_emb_dim = args.word_emb_dim
        self.use_cnn_fea = args.use_cnn_fea
        self.cnn_feature = CNN_BASE(args,pretrain_word_embedding,label_size,args.cnn_mode)
        self.cnn_dim = args.cnn_dim
        self.mode = args.model_classes
        self.cnn_pooling_mode = args.cnn_pooling_mode
        
        self.use_adv = args.use_adv
        self.adv_func = adversarial_train_loss(self._adv_forward)
        self.adv_loss_type = args.adv_loss_type

        self.num_token = 2 + args.use_multi_token_mtl
        self.hidden2token = nn.Linear(args.rnn_hidden_dim * 2,self.num_token)
        
        input_dim = self.word_emb_dim
        if self.mode == 'bilstm_cnn_pool':
            input_dim += self.cnn_dim
        
        self.lstm = nn.LSTM(input_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(input_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                          bidirectional=True)

        self.label_size = label_size
        if self.use_crf:
            self.crf = CRF(self.label_size, self.gpu)
            self.label_size += 2
        if self.use_highway:
            self.highway = Highway(args.rnn_hidden_dim * 2, 1)

        self.num_token = 2 + args.use_multi_token_mtl

        if self.use_cnn_fea:
            self.hidden2tag = nn.Linear(args.rnn_hidden_dim * 2 + self.cnn_dim, self.label_size)
            self.hidden2token = nn.Linear(args.rnn_hidden_dim * 2 + self.cnn_dim,self.num_token)
        else:
            self.hidden2tag = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)
            self.hidden2token = nn.Linear(args.rnn_hidden_dim * 2,self.num_token)


    def forward(self, word_input, input_mask, labels,labels_token=None,mode=None,model=None):
        # word_input input_mask   FloatTensor
        word_input_id = self.wordrep(word_input)

        input_mask.requires_grad = False
        word_input_id = word_input_id * (input_mask.unsqueeze(-1).float())

        batch_size = word_input_id.size(0)


        if self.mode == 'bilstm_cnn_pool':
            cnn_features = self.cnn_feature.get_output(word_input_id,input_mask,self.cnn_pooling_mode)
            cnn_features = cnn_features.expand((batch_size,word_input_id.size(1),word_input_id.size(2)))

            word_fea = torch.cat((word_input_id,cnn_features),-1)

            if self.rnn_type == 'LSTM':
                output, _ = self.lstm(word_fea)
            elif self.rnn_type == 'GRU':
                output, _ = self.gru(word_fea)

            if self.use_highway:
                output = self.highway(output)
        elif self.mode == 'bilstm_cnn':
            if self.rnn_type == 'LSTM':
                output, _ = self.lstm(word_input_id)
            elif self.rnn_type == 'GRU':
                output, _ = self.gru(word_input_id)

            if self.use_highway:
                output = self.highway(output)
            
            if self.use_cnn_fea:
                cnn_features = self.cnn_feature.get_output(word_input_id,input_mask)
                output = torch.cat((output,cnn_features),-1)

        output = self.dropoutlstm(output)
        output2 = self.hidden2tag(output)
        
        if labels_token is not None:
            if self.args.use_multi_token_mtl >=0 :
                token_output = self.hidden2token(output)
                loss_token = nn.CrossEntropyLoss()
                active_loss = input_mask.view(-1) == 1
                active_logits = token_output.view(-1, self.num_token)[active_loss]
                active_labels = labels_token.view(-1)[active_loss]
                token_loss = loss_token(active_logits, active_labels)

        maskk = input_mask.ge(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output2, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output2, input_mask)
            total_loss = total_loss / batch_size
            if self.args.use_multi_token_mtl>=0:
                    total_loss = total_loss + self.multi_token_loss_alpha * token_loss

            if self.use_adv and not mode:
                if self.adv_loss_type == 'fgm':
                    adv_loss = self.adv_func.adversarial_loss(total_loss,word_input_id,input_mask,labels,labels_token=labels_token)
                elif self.adv_loss_type == 'vat':
                    adv_loss = self.adv_func.virtual_adversarial_loss(output2,word_input_id,input_mask)
                elif self.adv_loss_type == 'pgd':
                    adv_loss = self.adv_func.PGD_loss(total_loss,word_input_id,input_mask,labels,labels_token=labels_token)
                elif self.adv_loss_type == 'fgm_vat':
                    adv_loss = self.adv_func.adversarial_loss(total_loss,word_input_id,input_mask,labels,labels_token=labels_token) \
                                    + self.adv_func.virtual_adversarial_loss(output2,word_input_id,input_mask)
                elif self.adv_loss_type == 'freelb':
                    adv_loss = 0
                    self.adv_func.Freelb_loss(model,word_input_id,input_mask,labels,labels_token)
                total_loss = total_loss  + adv_loss

            return total_loss, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)

            active_loss = input_mask.view(-1) == 1
            active_logits = output2.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            return loss, output

    def _adv_forward(self, word_input, input_mask, labels=None,labels_token=None):
        input_mask.requires_grad = False
        word_input_id = word_input * (input_mask.unsqueeze(-1).float())
        batch_size = word_input.size(0)

        if self.mode == 'bilstm_cnn_pool':
            cnn_features = self.cnn_feature.get_output(word_input_id,input_mask,self.cnn_pooling_mode)
            cnn_features = cnn_features.expand((batch_size,word_input_id.size(1),word_input_id.size(2)))

            word_fea = torch.cat((word_input_id,cnn_features),-1)

            if self.rnn_type == 'LSTM':
                output, _ = self.lstm(word_fea)
            elif self.rnn_type == 'GRU':
                output, _ = self.gru(word_fea)

            if self.use_highway:
                output = self.highway(output)
        elif self.mode == 'bilstm_cnn':
            if self.rnn_type == 'LSTM':
                output, _ = self.lstm(word_input_id)
            elif self.rnn_type == 'GRU':
                output, _ = self.gru(word_input_id)

            if self.use_highway:
                output = self.highway(output)
            
            if self.use_cnn_fea:
                cnn_features = self.cnn_feature.get_output(word_input_id,input_mask)
                output = torch.cat((output,cnn_features),-1)

        output = self.dropoutlstm(output)
        output2 = self.hidden2tag(output)

        if labels_token is not None:
            if self.args.use_multi_token_mtl>=0:
                token_output = self.hidden2token(output)
                loss_token = nn.CrossEntropyLoss(ignore_index=0)
                active_loss = input_mask.view(-1) == 1
                active_logits = token_output.view(-1, self.num_token)[active_loss]
                active_labels = labels_token.view(-1)[active_loss]
                token_loss = loss_token(active_logits, active_labels)

        maskk = input_mask.ge(1)
        if  labels is None:
            return output2
        else:
            if self.use_crf:
                total_loss = self.crf.neg_log_likelihood_loss(output2, maskk, labels)
                total_loss = total_loss / batch_size
                if self.args.use_multi_token_mtl>=0:
                        total_loss = total_loss + self.multi_token_loss_alpha * token_loss
                return total_loss
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)

                active_loss = input_mask.view(-1) == 1
                active_logits = output.view(-1, self.label_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

                return loss, output