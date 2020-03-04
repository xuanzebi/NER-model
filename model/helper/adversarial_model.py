from torch.autograd import grad
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def _scale_l2(x, epsilon):
    """
    被实验证明没啥用?
    x: [bsz, seq_len, dim]
    Divide x by max(abs(x)) for a numerically stable L2 norm.
    Scale over the full sequence, dims (1, 2)
    """
    bsz = x.size()[0]
    alpha = torch.max(torch.abs(x.reshape(bsz, -1)), dim=-1, keepdim=True).values + 1e-12
    alpha = alpha.unsqueeze(-1)
    l2_norm = alpha * torch.sqrt(
        torch.sum(
            torch.pow(x / alpha, 2),
            [1, 2], keepdim=True
        ) + 1e-6
    )
    x_unit = x / l2_norm
    return epsilon * x_unit

def _kl_divergence_with_logits(q_logits, p_logits):
    """Returns weighted KL divergence between distributions q and p.
    Args:
        q_logits: logits for 1st argument of KL divergence shape
              [batch_size, num_timesteps, num_classes] if num_classes > 2, and
              [batch_size, num_timesteps] if num_classes == 2.目标分布
        p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
    Returns:
        KL: float scalar.
    """
    q = F.softmax(q_logits, dim=-1)
    kl = torch.sum(
        q * (F.log_softmax(q_logits, -1) - F.log_softmax(p_logits, -1)),
        dim=-1
    )
    kl_loss = torch.mean(kl)
    return kl_loss

class adversarial_train_loss(object):
    def __init__(self, loss_fn,at_fgm_epsilon=1.0,vat_epsilon=1.0,vat_num_power_iteration=1,
                vat_small_constant_for_finite_diff=1e-1):
        super(adversarial_train_loss, self).__init__()
        self.loss_fn = loss_fn
        self.at_fgm_epsilon = at_fgm_epsilon
        self.vat_epsilon = vat_epsilon 
        self.vat_num_power_iteration = vat_num_power_iteration 
        self.vat_small_constant_for_finite_diff=vat_small_constant_for_finite_diff
    
    def adversarial_loss(self,loss,embedded,input_mask,labels,labels_token=None):
        emb_grad = grad(loss,embedded, retain_graph=True)[0]
        perturb = _scale_l2(emb_grad.detach(),self.at_fgm_epsilon)
        return self.loss_fn(embedded + perturb,input_mask,labels,labels_token)

    def virtual_adversarial_loss(self,logits, embedded,input_mask):
        logits4vat = logits.detach()
        # Initialize perturbation with random noise.
        d = torch.randn_like(embedded, requires_grad=True)
        for _ in range(self.vat_num_power_iteration):
            d.requires_grad = True
            d = d * input_mask.unsqueeze(-1).float()
            d = _scale_l2(d, self.vat_small_constant_for_finite_diff)

            vat_logits = self.loss_fn(embedded + d)

            kl = _kl_divergence_with_logits(logits4vat, vat_logits)
            d = grad(kl, d, retain_graph=True)[0]
            d = d.detach()

        vat_perb = _scale_l2(d, self.vat_epsilon)
        vat_logits_real = self.loss_fn(embedded + vat_perb)
        kl_loss = _kl_divergence_with_logits(logits4vat, vat_logits_real)

        return kl_loss
    
    # def combo_loss(self):
    #     return self.adversarial_loss() + self.virtual_adversarial_loss()

    def PGD_loss(self,):
        pass

    def Freelb_loss(self,):
        pass


def adversarial_loss(embedded, loss, loss_fn,input_mask,labels,perturb_norm_length=1.0,labels_token=None):
    """Adds gradient to embedding and recomputes classification loss.
        embedded: embedding参数    
        loss_fn: model 计算loss
    """
    # retain_graph True 梯度会在当前计算图累积
    emb_grad = grad(loss, embedded, retain_graph=True)[0]
    perturb = _scale_l2(emb_grad.detach(),perturb_norm_length)
    return loss_fn(embedded + perturb,input_mask,labels,labels_token)


class Bilstmcrf_adv(nn.Module):
    """
    bilstm-crf模型
    """

    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Bilstmcrf_adv, self).__init__()
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

        self.FGM = args.FGM
        self.adv_loss = adversarial_train_loss(self._adv_forward)
        self.adv_loss_type = args.adv_loss_type

        self.args = args
        self.word_emb_dim = args.word_emb_dim

        self.lstm = nn.LSTM(self.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(self.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                          bidirectional=True)

        self.label_size = label_size
        if self.use_crf:
            self.crf = CRF(self.label_size, self.gpu)
            self.label_size += 2
        self.hidden2tag = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)
        self.hidden2token = nn.Linear(args.rnn_hidden_dim * 2,2)
   
    def forward(self, word_input, input_mask, labels,labels_token=None,mode=None):
        # word_input input_mask   FloatTensor
        word_input_id = self.wordrep(word_input)

        input_mask.requires_grad = False
        word_input_id = word_input_id * (input_mask.unsqueeze(-1).float())

        batch_size = word_input_id.size(0)
        if self.rnn_type == 'LSTM':
            output, _ = self.lstm(word_input_id)
        elif self.rnn_type == 'GRU':
            output, _ = self.gru(word_input_id)

        output = self.dropoutlstm(output)

        output2 = self.hidden2tag(output)

        if self.args.use_token_mtl:
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

            total_loss = total_loss / batch_size
            if self.args.use_token_mtl:
                    total_loss = total_loss + token_loss

            if self.FGM and not mode:
                adv_loss = adversarial_loss(word_input_id,total_loss,self._adv_forward,input_mask,labels,labels_token=labels_token)
                total_loss = total_loss  + adv_loss
            return total_loss, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)

            active_loss = input_mask.view(-1) == 1
            active_logits = output.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            return loss, output

       
    def _adv_forward(self, word_input, input_mask, labels,labels_token):
        input_mask.requires_grad = False
        word_input_id = word_input * (input_mask.unsqueeze(-1).float())

        batch_size = word_input_id.size(0)
        if self.rnn_type == 'LSTM':
            output, _ = self.lstm(word_input_id)
        elif self.rnn_type == 'GRU':
            output, _ = self.gru(word_input_id)

        if self.use_highway:
            output = self.highway(output)

        output = self.dropoutlstm(output)
        output2 = self.hidden2tag(output)

        if self.args.use_token_mtl:
            token_output = self.hidden2token(output)
            loss_token = nn.CrossEntropyLoss(ignore_index=0)
            active_loss = input_mask.view(-1) == 1
            active_logits = token_output.view(-1, 2)[active_loss]
            active_labels = labels_token.view(-1)[active_loss]
            token_loss = loss_token(active_logits, active_labels)

        maskk = input_mask.ge(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output2, maskk, labels)
            total_loss = total_loss / batch_size
            if self.args.use_token_mtl:
                    total_loss = total_loss + token_loss
            return total_loss
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)

            active_loss = input_mask.view(-1) == 1
            active_logits = output.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            return loss, output