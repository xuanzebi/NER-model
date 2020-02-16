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

import sys

package_dir_b = "/opt/hyp/NER/NER-model"
sys.path.insert(0, package_dir_b)

import warnings

warnings.filterwarnings("ignore")
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from util.util import get_logger, compute_f1, compute_spans_bio, compute_spans_bieos, compute_instance_f1
from model.lstm.lstmcrf import Bilstmcrf
from model.helper.get_data import get_cyber_data, pregress


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate_instance(y_true, y_pred):
    metric = compute_instance_f1(y_true, y_pred)
    return metric


def evaluate_crf(y_true, y_pred, tag):
    if tag == 'BIO':
        gold_sentences = [compute_spans_bio(i) for i in y_true]
        pred_sentences = [compute_spans_bio(i) for i in y_pred]
    elif tag == 'BIEOS':
        gold_sentences = [compute_spans_bieos(i) for i in y_true]
        pred_sentences = [compute_spans_bieos(i) for i in y_pred]
    metric = compute_f1(gold_sentences, pred_sentences)
    return metric


def evaluate(data, model, label_map, tag, args, train_logger, device, dev_test_data, mode):
    print("Evaluating on {} set...".format(mode))
    test_iterator = tqdm(data, desc="dev_test_interation")
    y_pred = []
    y_true = []
    test_loss = 0.

    for step, test_batch in enumerate(test_iterator):
        print(len(test_batch))
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)
        input_ids, input_mask, label_ids = _test_batch

        # loss, logits = model.module.calculate_loss(input_ids, input_mask, label_ids) # if DataParallel model.module
        loss, logits = model(input_ids, input_mask, label_ids)

        if args.use_dataParallel:
            loss = torch.sum(loss)  # if DataParallel model.module

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
    metric['test_loss'] = test_loss / len(data)
    if mode == 'test':
        return metric, metric_instance, y_pred
    else:
        return metric, metric_instance


def train(model, train_dataloader, dev_dataloader, args, device, tb_writer, label_map, tag, train_logger,
          dev_test_data):
    # param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
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
    # save_model_list = [0,0,0,0,0]
    tr_loss, logging_loss = 0.0, 0.0
    lr = defaultdict(list)
    global_step = 0
    tq = tqdm(range(args.num_train_epochs), desc="Epoch")

    for epoch in tq:
        avg_loss = 0.
        model.train()
        model.zero_grad()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.zero_grad()
            _batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = _batch

            # loss, _ = model.module.calculate_loss(input_ids, input_mask, label_ids)
            loss, _ = model(input_ids, input_mask, label_ids)

            if args.use_dataParallel:
                loss = torch.sum(loss)  # if DataParallel

            tr_loss += loss.item()
            avg_loss += loss.item() / len(train_dataloader)

            loss.backward()
            optimizer.step()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.use_scheduler:
                    # a = scheduler.get_lr()
                    print(scheduler.get_lr()[0])
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    lr[epoch].append(scheduler.get_lr()[0])
                else:
                    for param_group in optimizer.param_groups:
                        lr[epoch].append(param_group['lr'])
                tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            epoch_iterator.set_postfix(train_loss=loss.item())

        if args.use_scheduler:
            scheduler.step()

        tq.set_postfix(avg_loss=avg_loss)

        print('%d epoch，global_step: %d ,train_loss: %.2f' % (epoch, global_step, tr_loss / global_step))
        train_logger.info('%d epoch，global_step: %d ,train_loss: %.2f' % (epoch, global_step, tr_loss / global_step))

        metric, metric_instance = evaluate(dev_dataloader, model, label_map, tag, args, train_logger, device,
                                           dev_test_data, 'dev')
        metric_instance['epoch'] = epoch
        metric['epoch'] = epoch

        print('epoch:{} P:{}, R:{}, F1:{}'.format(epoch, metric['precision-overall'], metric['recall-overall'],
                                                  metric['f1-measure-overall']))
        train_logger.info(
            'epoch:{} P:{}, R:{}, F1:{}'.format(epoch, metric['precision-overall'], metric['recall-overall'],
                                                metric['f1-measure-overall']))
        # print(metric['test_loss'], epoch)
        # train_logger.info("epoch{},test_loss{}".format(metric['test_loss'], epoch))

        tb_writer.add_scalar('test_loss', metric['test_loss'], epoch)

        if metric['micro-f1'] > bestscore:
            bestscore = metric['micro-f1']
            best_epoch = epoch
            print('实体级别的F1的best model epoch is: %d' % epoch)
            train_logger.info('实体级别的F1的best model epoch is: %d' % epoch)
            model_name = args.model_save_dir + "entity_best.pt"
            torch.save(model.state_dict(), model_name)

        # releax-f1 token-level f1
        if metric_instance['micro-f1'] > bestscore_instance:
            bestscore_instance = metric_instance['micro-f1']
            best_epoch_instance = epoch
            # print('token级别的F1best model epoch is: %d' % epoch)
            # train_logger.info('token级别的F1best model epoch is: %d' % epoch)
            # model_name = args.model_save_dir + "token_best.pt"
            # torch.save(model.state_dict(), model_name)

        test_result.append(metric)
        test_result_instance.append(metric_instance)

    test_result.append({'best_test_f1': bestscore,
                        'best_test_epoch': best_epoch})
    test_result_instance.append({'best_test_f1': bestscore_instance,
                                 'best_test_epoch': best_epoch_instance})
    tb_writer.close()
    return test_result, test_result_instance, lr


def load_predict(model, data, model_save_dir, logger, label_map, tag, args, device, test_data):
    start_time = time.time()
    model.load_state_dict(torch.load(model_save_dir))
    metric, metric_instance, y_pred = evaluate(data, model, label_map, tag, args, logger, device, test_data, 'test')
    end_time = time.time()
    print('预测Time Cost{}s'.format(end_time - start_time))
    logger.info('预测Time Cost{}s'.format(end_time - start_time))

    return metric, metric_instance, y_pred


def save_config(config, path, verbose=True):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config


if __name__ == "__main__":
    print(os.getcwd())
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_test", default=True, type=str2bool, help="Whether to run test on the test set.")
    parser.add_argument('--save_best_model', type=str2bool, default=False, help='Whether to save best model.')
    parser.add_argument('--model_save_dir', type=str, default='/opt/hyp/NER/NER-model/saved_models/test_msra/',
                        help='Root dir for saving models.')
    parser.add_argument('--data_path', default='/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data', type=str,
                        help='数据路径')
    parser.add_argument('--pred_embed_path', default='/opt/hyp/NER/embedding/sgns.baidubaike.bigram-char', type=str,
                        help="预训练词向量路径,'cc.zh.300.vec','sgns.baidubaike.bigram-char','Tencent_AILab_ChineseEmbedding.txt'")
    parser.add_argument('--tensorboard_dir', default='/opt/hyp/NER/NER-model/saved_models/test_msra/runs/', type=str)
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
    parser.add_argument('--deal_long_short_data', default='cut', choices=['cut', 'pad', 'stay'], type=str,
                        help='对长文本或者短文本在验证测试的时候如何处理')
    parser.add_argument('--save_embed_path',
                        default='/opt/hyp/NER/NER-model/data/embedding/sgns.baidubaike.bigram-char_msra.p', type=str,
                        help='词向量存储路径')

    # parser.add_argument('--data_type', default='conll', help='数据类型 -conll - cyber')
    parser.add_argument("--use_bieos", default=True, type=str2bool, help="True:BIEOS False:BIO")
    parser.add_argument('--token_level_f1', default=False, type=str2bool, help='Sequence max_length.')
    parser.add_argument('--do_lower_case', default=False, type=str2bool, help='False 不计算token-level f1，true 计算')
    parser.add_argument('--freeze', default=True, type=str2bool, help='是否冻结词向量')
    parser.add_argument('--use_crf', default=True, type=str2bool, help='是否使用crf')
    parser.add_argument('--rnn_type', default='LSTM', type=str, help='LSTM/GRU')
    parser.add_argument('--gpu', default=torch.cuda.is_available(), type=str2bool)
    parser.add_argument('--use_number_norm', default=False, type=str2bool)
    parser.add_argument('--use_pre', default=True, type=str2bool, help='是否使用预训练的词向量')
    parser.add_argument('--use_dataParallel', default=False, type=str2bool, help='是否使用dataParallel并行训练')
    parser.add_argument('--use_char', default=False, type=str2bool, help='是否使用char向量')
    parser.add_argument('--use_scheduler', default=True, type=str2bool, help='学习率是否下降')
    parser.add_argument('--load', default=True, type=str2bool, help='是否加载事先保存好的词向量')
    parser.add_argument('--use_highway', default=False, type=str2bool)
    parser.add_argument('--dump_embedding', default=False, type=str2bool, help='是否保存词向量')

    parser.add_argument("--learning_rate", default=0.015, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_seq_length', default=200, type=int, help='Sequence max_length.')
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--word_emb_dim', default=300, type=int, help='预训练词向量的维度')
    parser.add_argument('--char_emb_dim', default=30, type=int)
    parser.add_argument('--rnn_hidden_dim', default=128, type=int, help='rnn的隐状态的大小')
    parser.add_argument('--num_layers', default=1, type=int, help='rnn中的层数')
    parser.add_argument('--lr_decay', default=0.05, type=float)
    parser.add_argument('--momentum', default=0, type=float, help="0 or 0.9")
    parser.add_argument('--min_count', default=1, type=int)
    parser.add_argument('--dropout', default=0.25, type=float, help='词向量后的dropout')
    parser.add_argument('--dropoutlstm', default=0.25, type=float, help='lstm后的dropout')

    args = parser.parse_args()

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    result_dir = args.model_save_dir + '/result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    print(args)
    train_logger = get_logger(args.model_save_dir + '/train_log.log')
    train_logger.info('各参数数值为{}'.format(args))
    start_time = time.time()


    def seed_everything(SEED):
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True


    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_bieos == True:
        tag = 'BIEOS'
    else:
        tag = 'BIO'

    train_data_raw = json.load(open(args.data_path + '/train_data.json', encoding='utf-8'))
    test_data_raw = json.load(open(args.data_path + '/test_data.json', encoding='utf-8'))
    dev_data_raw = json.load(open(args.data_path + '/dev_data.json', encoding='utf-8'))
    train_logger.info('训练集大小为{}，验证集大小为{}，测试集大小为{}'.format(len(train_data_raw), len(dev_data_raw), len(test_data_raw)))

    new_data = []
    new_data.extend(train_data_raw)
    new_data.extend(test_data_raw)
    new_data.extend(dev_data_raw)

    pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label = get_cyber_data(new_data, args)

    args.vocab_size = len(vocab)

    train_data_id, train_mask_id, train_label_id = pregress(train_data_raw, word2idx, label2index,
                                                            max_seq_lenth=args.max_seq_length)
    train_data = torch.tensor([f for f in train_data_id], dtype=torch.long)
    train_mask = torch.tensor([f for f in train_mask_id], dtype=torch.long)
    train_label = torch.tensor([f for f in train_label_id], dtype=torch.long)
    train_dataset = TensorDataset(train_data, train_mask, train_label)

    dev_data, dev_mask, dev_label = pregress(dev_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
    dev_data = torch.tensor([f for f in dev_data], dtype=torch.long)
    dev_mask = torch.tensor([f for f in dev_mask], dtype=torch.long)
    dev_label = torch.tensor([f for f in dev_label], dtype=torch.long)
    dev_dataset = TensorDataset(dev_data, dev_mask, dev_label)

    test_data, test_mask, test_label = pregress(test_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
    test_data = torch.tensor([f for f in test_data], dtype=torch.long)
    test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
    test_label = torch.tensor([f for f in test_label], dtype=torch.long)
    test_dataset = TensorDataset(test_data, test_mask, test_label)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Bilstmcrf(args, pretrain_word_embedding, len(label2index))
    # model = nn.DataParallel(model.cuda())
    model = model.to(device)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_logger.info("Let's use{}GPUS".format(torch.cuda.device_count()))

    tb_writer = SummaryWriter(args.tensorboard_dir)

    if args.do_train:
        print('===============================开始训练================================')
        dev_result, dev_result_instance, lr = train(model, train_dataloader, dev_dataloader, args, device, tb_writer, \
                                                    index2label, tag, train_logger, dev_data_raw)

        with codecs.open(result_dir + '/dev_result.txt', 'w', encoding='utf-8') as f:
            json.dump(dev_result, f, indent=4, ensure_ascii=False)

        # with codecs.open(result_dir + '/dev_result_instance.txt', 'w', encoding='utf-8') as f:
        #     json.dump(dev_result_instance, f, indent=4, ensure_ascii=False)

        with codecs.open(args.model_save_dir + '/learning_rate.txt', 'w', encoding='utf-8') as f:
            json.dump(lr, f, indent=4, ensure_ascii=False)

        print(time.time() - start_time)

        opt = vars(args)  # dict
        # save config
        opt["time's"] = time.time() - start_time
        save_config(opt, args.model_save_dir + '/args_config.json', verbose=True)
        train_logger.info("Train Time cost{}min".format((time.time() - start_time) / 60))

    if args.do_test:
        print('=========================测试集==========================')
        print(args)
        test_model = Bilstmcrf(args, pretrain_word_embedding, len(label2index))
        test_model = test_model.to(device)
        # token_model_save_dir = args.model_save_dir + 'token_best.pt'
        entity_model_save_dir = args.model_save_dir + 'entity_best.pt'

        # token_metric,token_metric_instance,y_pred_token = load_predict(test_model,test_dataloader,token_model_save_dir,train_logger,index2label,tag,args,device)
        entity_metric, entity_metric_instance, y_pred_entity = load_predict(test_model, test_dataloader,
                                                                            entity_model_save_dir, \
                                                                            train_logger, index2label, tag, args,
                                                                            device, test_data_raw)

        # with codecs.open(result_dir + '/test_result_tokenmodel.txt', 'w', encoding='utf-8') as f:
        #     json.dump(token_metric, f, indent=4, ensure_ascii=False)
        # with codecs.open(result_dir + '/test_result_instance_tokenmodel.txt', 'w', encoding='utf-8') as f:
        #     json.dump(token_metric_instance, f, indent=4, ensure_ascii=False)

        with codecs.open(result_dir + '/test_result_entitymodel.txt', 'w', encoding='utf-8') as f:
            json.dump(entity_metric, f, indent=4, ensure_ascii=False)
        # with codecs.open(result_dir + '/test_result_instance_entitymodel.txt', 'w', encoding='utf-8') as f:
        #     json.dump(entity_metric_instance, f, indent=4, ensure_ascii=False)        

        # print(len(y_pred_entity))
        # print(len(test_data_raw))
        assert len(y_pred_entity) == len(test_data_raw)
        results = []
        for i, (text, label) in enumerate(test_data_raw):
            res = []
            res.append(text)
            res.append(label)
            res.append(' '.join(y_pred_entity[i]))
            results.append(res)

        with codecs.open(result_dir + '/test_pred_entity.txt', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
