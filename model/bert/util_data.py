from transformers import BertTokenizer


def text_tokenizer(orig_tokens, tokenizer):
    tokenizer = BertTokenizer.from_pretrained('D:/projects/nlp/bert/chinese_12_768_pytorch', do_lower_case=True)
    bert_tokens = []  # output
    for orig_token in orig_tokens:
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    return bert_tokens


# print(text_tokenizer(['你好', '==', 'ha'],tokenizer))

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_mask = label_mask


class First_token_replace_word:
    def __init__(self, max_seq_length, tokenizer, label_map):
        super(First_token_replace_word, self).__init__()
        self.max_seq_lenth = max_seq_length
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.longer = 0
        self.sen_len = []

    def _tokenize(self, tokens, labels):
        new_tokens = []
        new_labels = []
        for i, (word, label) in enumerate(zip(tokens, labels)):
            token = self.tokenizer.tokenize(word)
            for j, tok in enumerate(token):
                if j == 0:
                    new_tokens.append(tok)
                    new_labels.append(label)

        assert len(new_labels) == len(new_tokens)
        return new_tokens, new_labels

    def prepregress(self, data, use_sep=True):
        features = []
        for _tokens, _labels in data:
            tokens, labels = self._tokenize(_tokens, _labels)

            # 如果只加CLS 则 -1，如果加SEP 则 -2
            self.sen_len.append(len(tokens))
            if use_sep == True:
                if len(tokens) > self.max_seq_lenth - 2:
                    tokens = tokens[0:(self.max_seq_lenth - 2)]
                    labels = labels[0:(self.max_seq_lenth - 2)]
                    self.longer += 1
            else:
                if len(tokens) > self.max_seq_lenth - 1:
                    tokens = tokens[0:(self.max_seq_lenth - 1)]
                    labels = labels[0:(self.max_seq_lenth - 1)]
                    self.longer += 1

            segment_ids = []
            label_ids = []
            ntokens = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            label_ids.append(self.label_map["[CLS]"])

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(self.label_map[labels[i]])

            if use_sep == True:
                ntokens.append("[SEP]")
                segment_ids.append(0)
                label_ids.append(self.label_map["[SEP]"])

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < self.max_seq_lenth:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                ntokens.append("[PAD]")

            assert len(input_ids) == self.max_seq_lenth
            assert len(input_mask) == self.max_seq_lenth
            assert len(segment_ids) == self.max_seq_lenth
            assert len(label_ids) == self.max_seq_lenth

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))

        print('超出最大长度的文本数量有：', self.longer)

        return features


def get_labels(data, use_sep=True):
    label2index = {}
    label = set()
    label2index['[PAD]'] = 0
    index = 1
    labels = [la for _, la in data]

    for lab in labels:
        for la in lab.split(' '):
            label.add(la)
            if la not in label2index:
                label2index[la] = index
                index += 1

    label2index['[CLS]'] = index

    if use_sep == True:
        label2index['[SEP]'] = index + 1

    return label2index, label
