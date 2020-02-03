import json
import codecs
from collections import defaultdict
from util.util import *

# 从data  token格式转换为json格式
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
                strings = line.split(' ')
                if len(strings) == 1:  # 有一处 NN 无标签
                    # print(strings)
                    strings.append('O')

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


def save_data(data, label, save_name):
    with codecs.open(save_name, 'w', encoding='utf-8') as fw:
        data_label = []
        for i in range(len(data)):
            data_str = ' '.join(data[i])
            # BI to BIEOS
            # print(i, label[i])

            # label_bieos = to_bioes(label[i])
            # label_str = ' '.join(label_bieos)

            label_str = ' '.join(label[i])

            # print(data_str, len(data_str))
            # print(data[i], len(data[i]))
            # print(label_bieos, len(label_bieos))
            assert len(data[i]) == len(label[i])
            # data_label[data_str] = label_str
            data_label.append([data_str, label_str])

        print(len(data_label))
        json.dump(data_label, fw, indent=4, ensure_ascii=False)


def toke_to_json():
    train_file = 'D:/dataset/ChineseNERdataset/ResumeNER/train.char.bmes'
    test_file = 'D:/dataset/ChineseNERdataset/ResumeNER/test.char.bmes'
    dev_file = 'D:/dataset/ChineseNERdataset/ResumeNER/dev.char.bmes'

    # train_file = 'D:/dataset/ChineseNERdataset/贵州大学安全NER/Data/train.txt'
    # dev_file = 'D:/dataset/ChineseNERdataset/贵州大学安全NER/Data/dev.txt'
    # test_file = 'D:/dataset/ChineseNERdataset/贵州大学安全NER/Data/test.txt'

    train_data, train_label = get_data(train_file)
    save_data(train_data, train_label, 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/train_data.json')

    dev_data, dev_label = get_data(dev_file)
    save_data(dev_data, dev_label, 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/dev_data.json')

    test_data, test_label = get_data(test_file)
    save_data(test_data, test_label, 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/test_data.json')

    print(len(train_data), len(dev_data), len(test_data))


# train_file = 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/train_data.json'
# test_file = 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/test_data.json'
# dev_file = 'D:/dataset/ChineseNERdataset/ResumeNER/json_data/dev_data.json'


train_file = 'D:/dataset/ChineseNERdataset/贵州大学安全NER/json_data/train_data.json'
dev_file = 'D:/dataset/ChineseNERdataset/贵州大学安全NER/json_data/dev_data.json'
test_file = 'D:/dataset/ChineseNERdataset/贵州大学安全NER/json_data/test_data.json'


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

        # 分析句子长度/平均长度/涵盖97%的长度


analyse_data(train_file, test_file, dev_file)
