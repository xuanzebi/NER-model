import json
import codecs
from collections import defaultdict

import sys
package_dir_b = "/opt/hyp/NER/NER-model"
sys.path.insert(0, package_dir_b)


from util.util import *
import numpy as np


# 从data  token格式转换为json格式
def get_data(file, type='Resume'):
    # for file in filess:
    print(file)
    word_sequences = list()
    tag_sequences = list()

    token_num = 0
    entity_token = 0

    tag_set = set()
    curr_words = list()
    curr_tags = list()

    with open(file, 'r', encoding='utf-8-sig') as fr:
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
                if type == 'Resume' or type == 'Cyber':
                    strings = line.split(' ')
                else:
                    strings = line.split('\t')

                if len(strings) == 1:  # 有一处 NN 无标签
                    # print(strings)
                    strings.append('O')
                if type == 'Weibo':
                    word = strings[0][0]
                else:
                    word = strings[0]

                tag = strings[-1]
                if tag == 'I-PE':
                    print(k)

                tag_set.add(tag)
                curr_words.append(word)
                curr_tags.append(tag)

                token_num += 1
                if tag != 'O':
                    entity_token += 1

            if k == len(data_lins) - 1:
                assert len(curr_tags) == len(curr_words)
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)

    print('句子数量：', len(word_sequences))
    print('token总数量为 {},其中不为O的token数量为{}'.format(token_num, entity_token))
    print(tag_set, len(tag_set))
    return word_sequences, tag_sequences


def save_data(data, label, save_name, To_bieos=True):
    with codecs.open(save_name, 'w', encoding='utf-8') as fw:
        data_label = []
        for i in range(len(data)):
            data_str = ' '.join(data[i])
            # BI to BIEOS
            # print(i, label[i])

            if To_bieos == True:
                label_bieos = to_bioes(label[i])
                label_str = ' '.join(label_bieos)
            else:
                label_str = ' '.join(label[i])

            # print(data_str, len(data_str))
            # print(data[i], len(data[i]))
            # print(label_bieos, len(label_bieos))
            assert len(data[i]) == len(label[i])
            # data_label[data_str] = label_str
            if len(data[i]) > 1:
                data_label.append([data_str, label_str])

        print(len(data_label))
        json.dump(data_label, fw, indent=4, ensure_ascii=False)


def toke_to_json(type=None):
    # Resume
    train_file = '/opt/hyp/NER/NER-model/data/other_data/MSRA/train.ner'
    test_file = '/opt/hyp/NER/NER-model/data/other_data/MSRA/test.ner'
    dev_file = '/opt/hyp/NER/NER-model/data/other_data/MSRA/dev.ner'

    # train_file = '/opt/hyp/NER/NER-model/data/train.txt'
    # dev_file = '/opt/hyp/NER/NER-model/data/dev.txt'
    # test_file = '/opt/hyp/NER/NER-model/data/test.txt'

    train_data, train_label = get_data(train_file)
    save_data(train_data, train_label, '/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data/train_data.json', False)

    dev_data, dev_label = get_data(dev_file)
    save_data(dev_data, dev_label, '/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data/dev_data.json', False)

    test_data, test_label = get_data(test_file)
    save_data(test_data, test_label, '/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data/test_data.json', False)

    print(len(train_data), len(dev_data), len(test_data))


# Resume data
# train_file = 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/train_data.json'
# test_file = 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/test_data.json'
# dev_file = 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/dev_data.json'

# 中文安全数据
train_file = '/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data/train_data.json'
dev_file = '/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data/dev_data.json'
test_file = '/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data/test_data.json'


def analyse_data(*files):
    for file in files:
        print(file)
        data = json.load(open(file, 'r', encoding='utf-8'))
        print(len(data))

        labels = [la for _, la in data]
        texts = [la for la, _ in data]

        # 分析 每一类的实体
        label_set = set()
        label_dict = defaultdict(int)
        for lab in labels:
            lab = lab.split(' ')
            for la in lab:
                if len(la) == 1:
                    label_set.add(la)
                else:
                    label_set.add(la[2:])
            # for i, la in enumerate(lab):
            #     if i > 1 and len(lab[i]) > 1 and len(lab[i - 1]) == 1 and len(lab[i + 1]) == 1: # 判断单个实体怎么标识
            #         print(la)
            for i, la in enumerate(lab):
                if la[0] == 'S' or la[0] == 'E':
                    label_dict[la[2:]] += 1
        print(label_set, len(label_set))
        entities_num = sum(label_dict.values())
        print(label_dict, entities_num)

        # 分析句子中不包含实体的数量
        entity_sentences = 0
        for lab in labels:
            lab = lab.split(' ')
            for la in lab:
                if la != 'O':
                    entity_sentences += 1
                    break
        print('在{}个句子中，有{}个句子中没有一个实体'.format(len(data), len(data) - entity_sentences))

        # 分析句子长度/平均长度/涵盖97%的长度
        len_sentence = [len(i.split(' ')) for i in labels]
        a, b = 0, 0
        for i, j in enumerate(len_sentence):
            if j > 200:
                a += 1
            if j == 1:
                b += 1
        len_sentence = np.array(len_sentence)
        print('句子平均长度为{}， 句子最小长度为{}， 句子最大长度为{}，句子占比98%的长度为{}'.format(np.mean(len_sentence), np.min(len_sentence),
                                                                     np.max(len_sentence),
                                                                     np.percentile(len_sentence, 98)))
        print(a, b)

        # 计算token数量
        token_data = 0
        for lab in labels:
            lab = lab.split(' ')
            token_data += len(lab)
        print('token数量为{}'.format(token_data))


toke_to_json()
analyse_data(train_file, test_file, dev_file)


def bmes_to_bies(*files):
    for file in files:
        data_label = []
        data = json.load(open(file, 'r', encoding='utf-8'))
        labels = [la for _, la in data]
        texts = [la for la, _ in data]

        for i, lab in enumerate(labels):
            label = ['I' + l[1:] if l[0] == 'M' else l for l in lab.split(' ')]
            data_label.append([texts[i], ' '.join(label)])

        with codecs.open(file, 'w', encoding='utf-8') as fw:
            json.dump(data_label, fw, indent=4, ensure_ascii=False)


# bmes_to_bies(train_file, test_file, dev_file)


def count_label():
    resume_train_file = 'D:/Paper_Shiyan/NER-Model/data/ResumeNER/json_data/train_data.json'
    resume_dev_file = 'D:/Paper_Shiyan/NER-Model/data/ResumeNER/json_data/dev_data.json'
    resume_test_file = 'D:/Paper_Shiyan/NER-Model/data/ResumeNER/json_data/test_data.json'

    cyber_train_file = 'D:/Paper_Shiyan/NER-Model/data/Cyberdata/json_data/train_data.json'
    cyber_dev_file = 'D:/Paper_Shiyan/NER-Model/data/Cyberdata/json_data/dev_data.json'
    cyber_test_file = 'D:/Paper_Shiyan/NER-Model/data/Cyberdata/json_data/test_data.json'

    weibo_train_file = 'D:/Paper_Shiyan/NER-Model/data/WeiboNER/json_data/train_data.json'
    weibo_dev_file = 'D:/Paper_Shiyan/NER-Model/data/WeiboNER/json_data/dev_data.json'
    weibo_test_file = 'D:/Paper_Shiyan/NER-Model/data/WeiboNER/json_data/test_data.json'

    msra_train_file = 'D:/Paper_Shiyan/NER-Model/data/MSRA/json_data/train_data.json'
    msra_dev_file = 'D:/Paper_Shiyan/NER-Model/data/MSRA/json_data/dev_data.json'
    msra_test_file = 'D:/Paper_Shiyan/NER-Model/data/MSRA/json_data/test_data.json'

    def label_count(*files, type):
        label_set = set()
        for file in files:
            data = json.load(open(file, 'r', encoding='utf-8'))
            labels = [la for _, la in data]

            for lab in labels:
                for la in lab.split(' '):
                    label_set.add(la)

        print('======' + type + '的label为======')
        print(label_set, len(label_set))

    label_count(resume_train_file, resume_test_file, resume_dev_file, type='Resume')
    label_count(weibo_train_file, weibo_test_file, weibo_dev_file, type='Weibo')
    label_count(msra_train_file, msra_test_file, msra_dev_file, type='MSRA')
    label_count(cyber_train_file, cyber_test_file, cyber_dev_file, type='Cyber')


# count_label()

# TODO: 将四个数据集的label的交集修改为一样label（在使用多任务学习时）
