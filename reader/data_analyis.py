import os
import json
import numpy as np
from collections import defaultdict
""" MalwareTextDB Data """

train_file = 'D:/dataset/MalwareTextDB-2.0/data/train/tokenized'
dev_file = 'D:/dataset/MalwareTextDB-2.0/data/dev/tokenized'
test1_file = 'D:/dataset/MalwareTextDB-2.0/data/test_1/tokenized'
test2_file = 'D:/dataset/MalwareTextDB-2.0/data/test_2/tokenized'
test3_file = 'D:/dataset/MalwareTextDB-2.0/data/test_3/tokenized'

"""
    三类实体 ： 'I-Action', 'B-Action', 'B-Modifier', 'I-Modifier', 'B-Entity', 'O', 'I-Entity'
    训练集 ：9435 个句子
    验证集：
"""
def get_data(*filess):
    for filee in filess:
        print(filee)
        word_sequences = list()
        tag_sequences = list()

        token_num = 0
        entity_token = 0
        entity_phrase = 0

        for root, dirs, files in os.walk(filee):
            tag_set = set()
            for file in files:
                data_file = os.path.join(filee, file)
                curr_words = list()
                curr_tags = list()
                with open(data_file, 'r', encoding='utf-8') as fr:
                    data_lins = fr.readlines()
                    for k, linee in enumerate(data_lins):
                        line = linee.strip()
                        if len(line) == 0 and k != len(
                                data_lins) - 1:  # 处理最后一行是一个空行还是两个空行。 最后一个空行的话，该行不算单独一行，只是上一行+'\n'
                            if len(curr_words) > 0:
                                assert len(curr_words) == len(curr_tags)
                                word_sequences.append(curr_words)

                                # BIO to BIEOS
                                curr_tags = to_bioes(curr_tags)

                                tag_sequences.append(curr_tags)
                                curr_words = list()
                                curr_tags = list()
                            continue

                        # if k == len(data_lins) - 1 and len(line) != 0:  # 官方这几个文件最后一句话没有算
                        #     print(file)

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

        compute_num_entity(word_sequences, tag_sequences)
        compute_O(word_sequences, tag_sequences)
    # print(tag_set)


def compute_num_entity(data, label):
    label_idx = {}
    for i, j in zip(data, label):
        for token in j:
            if token == 'O':
                tok = token
            else:
                tok = token[2:]

            if tok in label_idx:
                if token[0] == 'S' or token[0] == 'E' or token[0] == 'O':
                    label_idx[tok] += 1
            else:
                if token[0] == 'S' or token[0] == 'E' or token[0] == 'O':
                    label_idx[tok] = 1
                else:
                    label_idx[tok] = 0

    num_entity = 0
    for i, j in label_idx.items():
        num_entity += j
    print('实体类别数量为：', label_idx)
    print('实体数量总数为:', num_entity)


# 计算全为0的数量和比例
def compute_O(data, label):
    num_O = 0
    for i, j in zip(data, label):
        flag = True
        for k in j:
            if k != 'O':
                flag = False
                break
        if flag == True:
            num_O += 1
    print('句子label全为O的数量为：', num_O)
    print('占比为:', num_O / len(data))


# 统计实体长度, 以及每类包含哪些实体
def get_entity(*files):
    for file in files:
        entity_type_num = defaultdict(dict)
        entity = [] 
        print(file)
        data = json.load(open(file, encoding='utf-8'))
        for text ,label in data:
            text = text.split(' ')
            label = label.split(' ')
            for i,la in enumerate(label):
                if la[0] == 'S':
                    entity.append(text[i])
                    if text[i] not in entity_type_num[la[2:]]:
                        entity_type_num[la[2:]][text[i]] = 1
                    else:
                        entity_type_num[la[2:]][text[i]] += 1
                if la[0] == 'B':
                    for j in range(i+1,len(label)):
                        if label[j][0] == 'E':
                            ent = ''.join(text[i:j+1])
                            entity.append(ent)
                            if ent not in entity_type_num[la[2:]]:
                                entity_type_num[la[2:]][ent] = 1
                            else:
                                entity_type_num[la[2:]][ent] += 1
                            break
    print(len(entity))
    return entity , entity_type_num

if __name__ == '__main__':
    """ 安全中文Data """
    train_file = '/opt/hyp/NER/NER-model/data/json_data/train_data.json'
    dev_file = '/opt/hyp/NER/NER-model/data/json_data/dev_data.json'
    test_file = '/opt/hyp/NER/NER-model/data/json_data/test_data.json'

    entity,entity_num = get_entity(train_file)
    a = {i:len(j) for i,j in entity_num.items()}
    print(a)

    sort_entnty_loc = sorted(entity_num['RT'].items(),key=lambda x: x[1],reverse=True)
    print(sort_entnty_loc[:10])
    # print(entity[:5])
    # print(entity_num)

    # 统计实体长度
    # entity_len = np.array([len(i) for i in entity])
    # print(np.max(entity_len),np.min(entity_len),np.mean(entity_len))
    # for i in entity:
    #     if len(i) > 10 :
    #         print(i)
    
    # 