import json
from model.helper.embedding import build_pretrain_embedding, check_coverage


# TODO 腾讯词向量 没有中文的， 《, 考虑将其换成中文的, 英文大写转小写
def build_vocab(data, min_count):
    """
        Return: vocab 词表各词出现的次数
                word2Idx 词表顺序
                label2index 标签对应序列
    """
    unk = '</UNK>'
    pad = '</PAD>'
    label2index = {}
    vocab = {}
    label2index[pad] = 0
    index = 1

    word2Idx = {}

    for i, line in enumerate(data):
        text = line[0].split(' ')
        label = line[1].split(' ')
        assert len(text) == len(label)
        for te, la in zip(text, label):
            word = te.strip()
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

            if la not in label2index:
                label2index[la] = index
                index += 1

    index2label = {j: i for i, j in label2index.items()}

    word2Idx[pad] = len(word2Idx)
    word2Idx[unk] = len(word2Idx)

    vocab = {i: j for i, j in vocab.items() if j >= min_count}

    for idx in vocab:
        if idx not in word2Idx:
            word2Idx[idx] = len(word2Idx)
    idx2word = {j: i for i, j in word2Idx.items()}

    return vocab, word2Idx, idx2word, label2index, index2label


def build_char_vocab(data):
    char2idx = {}
    for i, line in enumerate(data):
        text = line[0]
        for te in text:
            for t in te:
                if t not in char2idx:
                    char2idx[t] = len(char2idx)
    return char2idx


def save_embedding():
    import json
    data_path = '/opt/hyp/NER/NER-model/data/json_data'
    train_data = json.load(open(data_path + '/train_data.json', encoding='utf-8'))
    test_data = json.load(open(data_path + '/test_data.json', encoding='utf-8'))
    dev_data = json.load(open(data_path + '/dev_data.json', encoding='utf-8'))

    new_data = []
    new_data.extend(train_data)
    new_data.extend(test_data)
    new_data.extend(dev_data)

    vocab, word2idx, idx2word, label2index, index2label = build_vocab(new_data, 1)
    pretrain_word_embedding, unk_words, embedding_index = build_pretrain_embedding(
        '/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt', word2idx)
    print(unk_words)
    unk = check_coverage(vocab, embedding_index)
    print(unk)


# save_embedding()

def get_cyber_data(data, args):
    vocab, word2idx, idx2word, label2index, index2label = build_vocab(data, args.min_count)
    pretrain_word_embedding = build_pretrain_embedding(args, word2idx)
    return pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label


def pregress(data, word2idx, label2idx, max_seq_lenth):
    INPUT_ID = []
    INPUT_MASK = []
    LABEL_ID = []
    for text, label in data:
        input_mask = []
        input_id = []
        label_id = []
        text = text.split(' ')
        label = label.split(' ')
        for te, la in zip(text, label):
            te = te.strip()
            if te in word2idx:
                input_id.append(word2idx[te])
            else:
                input_id.append(word2idx['</UNK>'])
            label_id.append(label2idx[la])
            input_mask.append(1)

        if len(input_id) > max_seq_lenth:
            input_id = input_id[:max_seq_lenth]
            label_id = label_id[:max_seq_lenth]
            input_mask = input_mask[:max_seq_lenth]

        while len(input_id) < max_seq_lenth:
            input_id.append(0)
            label_id.append(0)
            input_mask.append(0)

        assert len(input_id) == len(label_id) == len(input_mask) == max_seq_lenth
        INPUT_ID.append(input_id)
        LABEL_ID.append(label_id)
        INPUT_MASK.append(input_mask)

    return INPUT_ID, INPUT_MASK, LABEL_ID

# 多任务学习，预测token是否为实体
def pregress_mtl(data, word2idx, label2idx, max_seq_lenth):
    INPUT_ID = []
    INPUT_MASK = []
    LABEL_ID = []
    TOKEN_LABEL_ID = []
    for text, label in data:
        input_mask = []
        input_id = []
        label_id = []
        token_id = []
        text = text.split(' ')
        label = label.split(' ')
        for te, la in zip(text, label):
            te = te.strip()
            if te in word2idx:
                input_id.append(word2idx[te])
            else:
                input_id.append(word2idx['</UNK>'])
            if la[0] == 'O':
                token_id.append(0)
            else:
                 token_id.append(1)
            label_id.append(label2idx[la])
            input_mask.append(1)

        if len(input_id) > max_seq_lenth:
            input_id = input_id[:max_seq_lenth]
            label_id = label_id[:max_seq_lenth]
            input_mask = input_mask[:max_seq_lenth]
            token_id = token_id[:max_seq_lenth]

        while len(input_id) < max_seq_lenth:
            input_id.append(0)
            label_id.append(0)
            input_mask.append(0)
            token_id.append(0)

        assert len(input_id) == len(label_id) == len(input_mask) == len(token_id) == max_seq_lenth
        INPUT_ID.append(input_id)
        LABEL_ID.append(label_id)
        INPUT_MASK.append(input_mask)
        TOKEN_LABEL_ID.append(token_id)

    return INPUT_ID, INPUT_MASK, LABEL_ID,TOKEN_LABEL_ID

