import os
import json
import codecs
from collections import defaultdict


def get_data(file):
    # for file in filess:
    print(file)
    word_sequences = list()
    tag_sequences = list()

    token_num = 0
    entity_token = 0
    entity_phrase = 0
    tag_set = set()
    curr_words = list()
    curr_tags = list()

    with open(file, 'r', encoding='utf-8') as fr:
        data_lins = fr.readlines()
        for k, linee in enumerate(data_lins):
            line = linee.strip()
            if len(line) == 0 and k != len(data_lins) - 1:  # 处理最后一行是一个空行还是两个空行。 最后一个空行的话，该行不算单独一行，只是上一行+'\n'
                if len(curr_words) > 0:
                    assert len(curr_words) == len(curr_tags)
                    word_sequences.append(curr_words)

                    tag_sequences.append(curr_tags)
                    curr_words = list()
                    curr_tags = list()
                continue

            if len(line) != 0:
                strings = line.split(' ')
                if len(strings) == 1:  # 有一处 NN 无标签
                    # print(strings)
                    strings.append('O')

                word = strings[0]
                tag = strings[1]
                tag_set.add(tag)
                curr_words.append(word)
                curr_tags.append(tag)

                token_num += 1
                if tag != 'O':
                    entity_token += 1

            if k == len(data_lins) - 1:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)

    print('句子数量：', len(word_sequences))
    print('token总数量为 {},其中不为O的token数量为{}'.format(token_num, entity_token))
    print(tag_set)
    return word_sequences, tag_sequences


train_file = 'D:/dataset/NER-master/Data/train.txt'
dev_file = 'D:/dataset/NER-master/Data/dev.txt'
test_file = 'D:/dataset/NER-master/Data/test.txt'


def save_data(data, label, save_name):
    with codecs.open(save_name, 'w', encoding='utf-8') as fw:
        data_label = defaultdict(list)
        for i in range(len(data)):
            data_str = ''.join(data[i])
            label_str = ' '.join(label[i])

            data_label[data_str] = label_str
        json.dump(data_label, fw, indent=4, ensure_ascii=False)


train_data, train_label = get_data(train_file)
save_data(train_data, train_label, 'train_data.json')

dev_data, dev_label = get_data(dev_file)
save_data(dev_data, dev_label, 'dev_data.json')

test_data, test_label = get_data(test_file)
save_data(test_data, test_label, 'test_data.json')