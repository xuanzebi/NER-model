import torch.nn as nn
import torch

# https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py
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