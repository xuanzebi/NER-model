import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_pretrain_embedding(embedding_path, word_index,word_dim):
    embedding_matrix = np.zeros((len(word_index), word_dim))
    alphabet_size = len(word_index)
    scale = np.sqrt(3 / word_dim)
    # index:0 padding
    for index in range(1, len(word_index)):
        embedding_matrix[index, :] = np.random.uniform(-scale, scale, [1, word_dim])
    
    perfect_match = 0
    case_match = 0
    not_match = 0

    if embedding_path == None or embedding_path == '':
        print('================不加载词向量================')
        return embedding_matrix, 0
    else:
        print('===============GLOVE===================')
        #embedding_index = load_embeddings(embedding_path)
        embedding_index,word_dim = load_pretrain_emb(embedding_path)
        unknown_words = []
        for word, i in word_index.items():
            if word in embedding_index:
                embedding_matrix[i] = embedding_index[word]
                perfect_match += 1
            elif word.lower in embedding_index:
                embedding_matrix[i] = embedding_index[word.lower()]
                case_match += 1
            elif word.title() in embedding_index:
                embedding_matrix[i] = embedding_index[word.title()]
                case_match += 1
            else:
                unknown_words.append(word)
                notmatch += 1

        unkword_set = set(unknown_words)
        pretrained_size = len(embedding_index)
        print("unk words 数量为{},unk 的word（set）数量为{}".format(len(unknown_words),len(unknown_words)))
        print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
        return embedding_matrix, unknown_words


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim



import operator 

# 检查oov
def check_coverage(vocab,embeddings_index):
    """
        vocab: 词表 字典形式
        embeddings_index: 加载的词向量
    """
    a = {}
    oov = {}
    k = 0
    i = 0
    perfect_match = 0
    case_match = 0
    not_match = 0
    title_match = 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except KeyError:
            try:
                a[word] = embeddings_index[word.lower()]
                case_match += 1
            except KeyError:
                try:
                    a[word] = embeddings_index[word.title()]
                    title_match += 1
                except KeyError:
                    oov[word] = vocab[word] # vocab[word] 代表该词在文本中出现了几次
                    i += vocab[word] 
                    pass

    # 词表的覆盖度  会有一些出现一次的词来降低覆盖度，所以需要设置最小频率
    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for %d of case match' %case_match)
    print('Found embeddings for %d of title match'%title_match)
    # 所有文本的覆盖度
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x   # 可以查看哪些词没有被覆盖到。


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def build_word2vec_embedding(model,word_index):
    embedding_matrix = np.zeros((len(word_index), 100))
    scale = np.sqrt(3.0 / 100)
    for index in range(1, len(word_index)):
        embedding_matrix[index, :] = np.random.uniform(-scale, scale, [1, 100])

    unknown_words = []
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = model.wv[word]
        except KeyError:
            try:
                embedding_matrix[i] = model.wv[word.lower()]
            except KeyError:
                try:
                    embedding_matrix[i] = model.wv[word.title()]
                except KeyError:
                    unknown_words.append(word)
    return embedding_matrix, unknown_words

# 统计词典覆盖度
