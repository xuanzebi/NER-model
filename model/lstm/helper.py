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


def train_mtl(model, train_dataloader, dev_dataloader, args, device, tb_writer, label_map, tag, train_logger,
          dev_test_data,other_data=None):
    # param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    train_loss_step = {}
    train_loss_epoch = {}
    dev_loss_epoch = {}

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-8)

    if args.use_scheduler:
        decay_rate = 0.05
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 / (1 + decay_rate * epoch))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    test_result = []
    test_result_instance = []
    bestscore, best_epoch = -1, 0
    bestscore_instance, best_epoch_instance = -1, 0
    save_model_list = [0,0,0,0,0]
    save_model_epoch= [-1,-1,-1,-1,-1]

    tr_loss, logging_loss = 0.0, 0.0
    lr = defaultdict(list)
    global_step = 0
    tq = tqdm(range(args.num_train_epochs), desc="Epoch")

    p = 0

    if args.use_fgm:
        fgm = FGM(model)

    for epoch in tq:
        avg_loss = 0.
        epoch_start_time = time.time()
        model.train()
        model.zero_grad()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch,msra_batch in enumerate(zip(epoch_iterator,other_data)):
            model.zero_grad()
            _batch = tuple(t.to(device) for t in batch)

            msra_batch = other_data[step]
            msra_batch = tuple(t.to(device) for t in msra_batch)
            # msra_input_ids, msra_input_mask, msra_label_ids,msra_token_id = msra_batch 
            # input_ids, input_mask, label_ids,token_id = _batch 
            loss,_ = model(_batch,msra_batch)
            loss.backward()
            optimizer.step()


            if args.use_dataParallel:
                loss = torch.sum(loss)  # if DataParallel

            tr_loss += loss.item()
            avg_loss += loss.item() / len(train_dataloader)
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.use_scheduler:
                    print('当前epoch {}, step{} 的学习率为{}'.format(epoch, step, scheduler.get_lr()[0]))
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    lr[epoch].append(scheduler.get_lr()[0])
                else:
                    for param_group in optimizer.param_groups:
                        lr[epoch].append(param_group['lr'])
                tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                train_loss_step[global_step] = (tr_loss - logging_loss) / args.logging_steps
                logging_loss = tr_loss

            epoch_iterator.set_postfix(train_loss=loss.item())

        if args.use_scheduler:
            scheduler.step()

        tq.set_postfix(avg_loss=avg_loss)
        train_loss_epoch[epoch] = avg_loss
        metric, metric_instance = evaluate(dev_dataloader, model, label_map, tag, args, train_logger, device,dev_test_data, 'dev')

        metric_instance['epoch'] = epoch
        metric['epoch'] = epoch
        dev_loss_epoch[epoch] = metric['test_loss']
        # print(metric['test_loss'], epoch)
        # train_logger.info("epoch{},test_loss{}".format(metric['test_loss'], epoch))

        tb_writer.add_scalar('test_loss', metric['test_loss'], epoch)

        if args.save_best_model:
            if metric['micro-f1'] > bestscore:
                bestscore = metric['micro-f1']
                best_epoch = epoch
                print('实体级别的F1的best model epoch is: %d' % epoch)
                train_logger.info('实体级别的F1的best model epoch is: %d' % epoch)
                model_name = args.model_save_dir + "entity_best.pt"
                torch.save(model.state_dict(), model_name)
        else:      # 保存最佳的5个模型，取平均
            if metric['micro-f1'] > min(save_model_list):
                low_index = save_model_list.index(min(save_model_list))
                save_model_list[low_index] =  metric['micro-f1'] 
                save_model_epoch[low_index] = epoch
                bestscore = max(save_model_list)
                p += 1
                model_name = args.model_save_dir + "entity_best." + str(p) + ".pt"
                torch.save(model.state_dict(), model_name)
                if p == 5:
                    p = 0

        print('epoch {} , global_step {}, train_loss {}, train_avg_loss:{}, dev_avg_loss:{}, 该epoch耗时:{}s!'.format(epoch, global_step, 
                                                     tr_loss / global_step,avg_loss,metric['test_loss'],time.time()-epoch_start_time))
        train_logger.info('epoch {} , global_step {}, train_loss {}, train_avg_loss:{}, dev_avg_loss:{},该epoch耗时:{}s!'.format(epoch, global_step, 
                                                            tr_loss / global_step,avg_loss,metric['test_loss'],time.time()-epoch_start_time))

        print('epoch:{} P:{}, R:{}, F1:{}, best F1:{}!"\n"'.format(epoch, metric['precision-overall'],
                                                              metric['recall-overall'],
                                                              metric['f1-measure-overall'], bestscore))
        train_logger.info(
            'epoch:{} P:{}, R:{}, F1:{}, best F1:{}!"\n"'.format(epoch, metric['precision-overall'], metric['recall-overall'],
                                                           metric['f1-measure-overall'], bestscore))

        test_result.append(metric)

    if args.save_best_model:
        test_result.append({'best_dev_f1': bestscore,
                            'best_dev_epoch': best_epoch})
    else:
        test_result.append({'best_dev_f1': bestscore,
                            'dev_bestof5_epoch': (save_model_list,save_model_epoch)})   

    tb_writer.close()
    return test_result, test_result_instance, lr, train_loss_step, train_loss_epoch,dev_loss_epoch