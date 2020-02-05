def build_vocab(data, min_count):
    unk = '</UNK>'
    pad = '</PAD>'
    label2index = {}
    vocab = {}
    label2index[pad] = 0
    index = 1

    word2Idx = {}

    for i, line in enumerate(data):
        text = line[0]
        label = line[1]
        for te, la in zip(text, label):
            if te in vocab:
                vocab[te] += 1
            else:
                vocab[te] = 1

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
