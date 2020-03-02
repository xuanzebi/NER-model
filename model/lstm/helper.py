import os
import argparse
import time
from collections import defaultdict
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import codecs
import json
from sklearn.model_selection import StratifiedKFold, KFold

from model.helper.evaluate import evaluate_crf,evaluate_instance,evaluate_st_end

def evaluate(data, model, label_map, tag, args, train_logger, device, dev_test_data, mode):
    print("Evaluating on {} set...".format(mode))
    test_iterator = tqdm(data, desc="dev_test_interation")
    y_pred = []
    y_true = []
    test_loss = 0.
    test_step = 0

    for step, test_batch in enumerate(test_iterator):
        # print(len(test_batch))
        test_step += 1
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)

        with torch.no_grad():
            if args.model_classes == 'bilstm':
                if args.use_bert or args.use_elmo:
                    input_ids, input_mask, label_ids,quid = _test_batch 
                    loss, logits = model(input_ids, input_mask, label_ids,quid,mode)
                else:
                    input_ids, input_mask, label_ids = _test_batch  
                    loss, logits = model(input_ids, input_mask, label_ids)
            elif args.model_classes == 'bilstm_mtl':
                input_ids, input_mask, label_ids,token_id = _test_batch  
                loss, logits = model(input_ids, input_mask, label_ids,token_id)
            elif args.model_classes == 'bilstm_data_mtl':
                input_ids, input_mask, label_ids,token_id,data_type = _test_batch 
                a = [int(k) for k in data_type]
                a = set(a)
                assert len(a) == 1  # 确保每个batch里只有一个data_type
                data_type_id = int(data_type[0])
                loss,logits=model(input_ids, input_mask, label_ids,token_id,data_type_id)

        if args.use_dataParallel:
            loss = torch.sum(loss)  # if DataParallel model.module  or torch.mean()

        if args.use_crf == False:
            logits = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)

        logits = logits.detach().cpu().numpy()
        test_loss += loss.item()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.cpu().data.numpy()

        if args.deal_long_short_data == 'cut':
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if input_mask[i][j] != 0:
                        temp_1.append(label_map[m])
                        temp_2.append(label_map[logits[i][j]])
                        if j == label.size - 1:  # len(label),args.max_seq_len
                            assert (len(temp_1) == len(temp_2))
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                    elif input_mask[i][j] == 0:
                        assert (len(temp_1) == len(temp_2))
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
        elif args.deal_long_short_data == 'pad':
            # 只保存到定义的max_seq_len的长度
            for i, label in enumerate(label_ids):
                y_true.append([label_map[m] for m in label])
                y_pred.append([label_map[m] for m in logits[i]])
        elif args.deal_long_short_data == 'stay':
            for i, label in enumerate(label_ids):
                dev_test_len = dev_test_data[i][1].split(' ')
                if len(dev_test_len) <= args.max_seq_length:
                    y_true.append([label_map[m] for m in label])
                    y_pred.append([label_map[m] for m in logits[i]])
                else:
                    tmp_pred = [label_map[m] for m in logits[i]]
                    tmp2 = len(dev_test_len) - args.max_seq_length
                    while tmp2 > 0:
                        tmp_pred.append('O')
                        tmp2 -= 1
                    y_true.append(dev_test_len)
                    y_pred.append(tmp_pred)
                    assert len(dev_test_len) == len(tmp_pred)

        test_iterator.set_postfix(test_loss=loss.item())

    metric_instance = evaluate_instance(y_true, y_pred)
    metric = evaluate_crf(y_true, y_pred, tag)
    metric['test_loss'] = test_loss / test_step
    if mode == 'test':
        return metric, metric_instance, y_pred
    else:
        return metric, metric_instance

