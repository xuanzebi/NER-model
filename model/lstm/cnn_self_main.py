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
import gc

import sys

package_dir_b = "/opt/hyp/NER/NER-model"
sys.path.insert(0, package_dir_b)

import warnings

warnings.filterwarnings("ignore")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.util import get_logger, compute_f1, compute_spans_bio, compute_spans_bieos, compute_instance_f1
from model.lstm.lstmcrf import Bilstmcrf
from model.lstm.model import Bilstm_CRF_MTL,Bilstm_ST_END,Bilstm_MTL
from model.helper.get_data import get_cyber_data, pregress,pregress_mtl,pregress_bert_embedding
from model.helper.convert_data import gen_token_ner_lstm, cys_label
from model.helper.evaluate import evaluate_crf,evaluate_instance,evaluate_st_end
from model.helper.batchsamper import MultitastdataBatchsampler
from model.helper.adversarial_model import Bilstmcrf_adv
from model.lstm.cnn_self_attn import CNN_BASE,Bilstmcrf_cnn

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
            if args.use_multi_token_mtl < 0:
                input_ids, input_mask, label_ids = _test_batch  
                loss, logits = model(input_ids, input_mask, label_ids,mode='dev')
            elif args.use_multi_token_mtl >= 0:
                input_ids, input_mask, label_ids,token_id = _test_batch  
                if args.use_adv:
                    loss, logits = model(input_ids, input_mask, label_ids,labels_token=token_id,mode='dev')
                else:
                    loss, logits = model(input_ids, input_mask, label_ids,labels_token=token_id)

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
        test_iterator.set_postfix(test_loss=loss.item())

    metric_instance = evaluate_instance(y_true, y_pred)
    metric = evaluate_crf(y_true, y_pred, tag)
    metric['test_loss'] = test_loss / test_step
    if mode == 'test':
        return metric, metric_instance, y_pred
    else:
        return metric, metric_instance

def train(model, train_dataloader, dev_dataloader, args, device, tb_writer, label_map, tag, train_logger,
          dev_test_data):
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

    for epoch in tq:
        avg_loss = 0.
        epoch_start_time = time.time()
        model.train()
        model.zero_grad()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.zero_grad()
            _batch = tuple(t.to(device) for t in batch)
            
            if args.use_multi_token_mtl < 0:
                input_ids, input_mask, label_ids = _batch 
                loss, _ = model(input_ids, input_mask, label_ids)
                loss.backward()
                optimizer.step()
            elif args.use_multi_token_mtl >= 0:
                input_ids, input_mask, label_ids,token_id = _batch 
                loss, _ = model(input_ids, input_mask, label_ids,token_id)
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
        test_result_instance.append({'best_dev_f1': bestscore_instance,
                                    'best_dev_epoch': best_epoch_instance})
    else:
        test_result.append({'best_dev_f1': bestscore,
                            'dev_bestof5_epoch': (save_model_list,save_model_epoch)})   

              
    tb_writer.close()
    return test_result, test_result_instance, lr, train_loss_step, train_loss_epoch,dev_loss_epoch


def load_predict(model, data, model_save_dir, logger, label_map, tag, args, device, test_data):
    start_time = time.time()
    model.load_state_dict(torch.load(model_save_dir))
    metric, metric_instance, y_pred = evaluate(data, model, label_map, tag, args, logger, device, test_data, 'test')
    end_time = time.time()
    print('预测Time Cost:{}s'.format(end_time - start_time))
    logger.info('预测Time Cost:{}s'.format(end_time - start_time))

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
    parser.add_argument("--do_test", default=False, type=str2bool, help="Whether to run test on the test set.")
    parser.add_argument('--model_classes', type=str, default='bilstm_cnn_pool',
                    choices=['bilstm','bilstm_mtl','bilstm_start_end',"bilstm_cnn","cnn","bilstm_cnn_pool"], help='Which model to choose.')
    parser.add_argument('--model_save_dir', type=str, default='/opt/hyp/NER/NER-model/saved_models/test/',
                        help='Root dir for saving models.')
    parser.add_argument('--data_path', default='/opt/hyp/NER/NER-model/data/other_data/ResumeNER/json_data', type=str,help='数据路径')
    parser.add_argument('--pred_embed_path', default='/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt', type=str,
                        help="预训练词向量路径,'cc.zh.300.vec','sgns.baidubaike.bigram-char','Tencent_AILab_ChineseEmbedding.txt'")
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
    parser.add_argument('--rnn_type', default='LSTM', type=str, help='LSTM/GRU')
    parser.add_argument('--adv_loss_type', default='pgd', choices=['','fgm','vat','pgd','freelb','fgm_vat'], type=str)
    parser.add_argument('--cnn_pooling_mode', default='seq', choices=['seq','dim'], type=str,help="dim 是在句子长度上max_pooling，seq是在每个词的所有特征上max_pooling")
    parser.add_argument('--cnn_mode', default='CNNPOOLING', choices=['CNN','GCNN','GLDR','CNNPOOLING'], type=str)
    parser.add_argument('--pos_emb', default=0, choices=[0,1,2], type=int)
    parser.add_argument('--deal_long_short_data', default='cut', choices=['cut', 'pad', 'stay'], type=str, help='对长文本或者短文本在验证测试的时候如何处理')
    parser.add_argument('--save_embed_path',default='/opt/hyp/NER/NER-model/data/embedding/Tencent_AILab_ChineseEmbedding_resume.p', type=str,
                        help='词向量存储路径')

    # parser.add_argument('--data_type', default='conll', help='数据类型 -conll - cyber')
    parser.add_argument("--use_bieos", default=True, type=str2bool, help="True:BIEOS False:BIO")
    parser.add_argument('--save_best_model', type=str2bool, default=False, help='Whether to save best model.')
    parser.add_argument('--token_level_f1', default=False, type=str2bool, help='Sequence max_length.')
    parser.add_argument('--do_lower_case', default=False, type=str2bool, help='False 不计算token-level f1，true 计算')
    parser.add_argument('--freeze', default=False, type=str2bool, help='是否冻结词向量')
    parser.add_argument('--msra_freeze', default=True, type=str2bool, help='多任务的词向量是否冻结词向量')
    parser.add_argument('--use_crf', default=True, type=str2bool, help='是否使用crf')
    parser.add_argument('--gpu', default=torch.cuda.is_available(), type=str2bool)
    parser.add_argument('--use_number_norm', default=False, type=str2bool)
    parser.add_argument('--use_pre', default=True, type=str2bool, help='是否使用预训练的词向量')
    parser.add_argument('--use_dataParallel', default=False, type=str2bool, help='是否使用dataParallel并行训练')
    parser.add_argument('--use_char', default=False, type=str2bool, help='是否使用char向量')
    parser.add_argument('--use_scheduler', default=True, type=str2bool, help='学习率是否下降')
    parser.add_argument('--load', default=True, type=str2bool, help='是否加载事先保存好的词向量')
    parser.add_argument('--use_highway', default=False, type=str2bool)
    parser.add_argument('--dump_embedding', default=False, type=str2bool, help='是否保存词向量')
    parser.add_argument('--use_packpad', default=False, type=str2bool, help='是否使用packed_pad')
    parser.add_argument('--use_adv', default=True, type=str2bool, help='是否使用对抗样本')
    parser.add_argument('--use_cnn_fea', default=False, type=str2bool, help='是否使用cnn')

    parser.add_argument("--learning_rate", default=0.015, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--char_emb_dim', default=30, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_seq_length', default=190, type=int, help='Sequence max_length.')
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--word_emb_dim', default=200, type=int, help='预训练词向量的维度')
    parser.add_argument('--cnn_dim', default=200, type=int, help='')
    parser.add_argument('--use_multi_token_mtl', default=-1, type=int, help='0/1/2  0表示不用，1表示使用3个token，2表示4个token')
    parser.add_argument('--multi_token_loss_alpha', default=-1, type=int, help='多任务学习的权重')
    # parser.add_argument('--gcnn_inter_dim', default=200, type=int, help='')
    parser.add_argument('--rnn_hidden_dim', default=128, type=int, help='rnn的隐状态的大小')
    parser.add_argument('--num_layers', default=1, type=int, help='rnn中的层数')
    parser.add_argument('--lr_decay', default=0.05, type=float)
    parser.add_argument('--momentum', default=0, type=float, help="0 or 0.9")
    parser.add_argument('--min_count', default=1, type=int)
    parser.add_argument('--dropout', default=0.2, type=float, help='词向量后的dropout')
    parser.add_argument('--dropoutlstm', default=0.2, type=float, help='lstm后的dropout')

    args = parser.parse_args()

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if args.do_train:
        if os.path.exists(args.model_save_dir):
            for root, dirs, files in os.walk(args.model_save_dir):
                for sub_dir in dirs:
                    for sub_root, sub_di, sub_files in os.walk(os.path.join(root,sub_dir)):
                        for sub_file in sub_files:
                            os.remove(os.path.join(sub_root,sub_file))
                for envent_file in files:
                    os.remove(os.path.join(root,envent_file))

    result_dir = args.model_save_dir + '/result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    args.tensorboard_dir = args.model_save_dir + '/runs'
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

    args.label = label2index
    # args.idx2word = idx2word
    args.vocab_size = len(vocab)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_logger.info("Let's use{}GPUS".format(torch.cuda.device_count()))

    tb_writer = SummaryWriter(args.tensorboard_dir)

    if args.do_train:
        args.test = False
        # Dataset
        if args.use_multi_token_mtl < 0:
            train_data_id, train_mask_id, train_label_id = pregress(train_data_raw, word2idx, label2index,max_seq_lenth=args.max_seq_length)
            train_data = torch.tensor([f for f in train_data_id], dtype=torch.long)
            train_mask = torch.tensor([f for f in train_mask_id], dtype=torch.long)
            train_label = torch.tensor([f for f in train_label_id], dtype=torch.long)
            train_dataset = TensorDataset(train_data, train_mask, train_label)

            dev_data, dev_mask, dev_label = pregress(dev_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
            dev_data = torch.tensor([f for f in dev_data], dtype=torch.long)
            dev_mask = torch.tensor([f for f in dev_mask], dtype=torch.long)
            dev_label = torch.tensor([f for f in dev_label], dtype=torch.long)
            dev_dataset = TensorDataset(dev_data, dev_mask, dev_label)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
        elif args.use_multi_token_mtl >= 0:
            train_data_id, train_mask_id, train_label_id,train_token_id = pregress_mtl(train_data_raw, word2idx, label2index,max_seq_lenth=args.max_seq_length,
                                                                                                                            mode=args.use_multi_token_mtl)
            train_data = torch.tensor([f for f in train_data_id], dtype=torch.long)
            train_mask = torch.tensor([f for f in train_mask_id], dtype=torch.long)
            train_label = torch.tensor([f for f in train_label_id], dtype=torch.long)
            train_token = torch.tensor([f for f in train_token_id], dtype=torch.long)            
            train_dataset = TensorDataset(train_data, train_mask, train_label,train_token)

            dev_data, dev_mask, dev_label,dev_token_id = pregress_mtl(dev_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length,mode=args.use_multi_token_mtl)
            dev_data = torch.tensor([f for f in dev_data], dtype=torch.long)
            dev_mask = torch.tensor([f for f in dev_mask], dtype=torch.long)
            dev_label = torch.tensor([f for f in dev_label], dtype=torch.long)
            dev_token = torch.tensor([f for f in dev_token_id], dtype=torch.long)
            dev_dataset = TensorDataset(dev_data, dev_mask, dev_label,dev_token)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

        # Model
        if args.model_classes == 'bilstm':
            if args.use_adv:
                model = Bilstmcrf_adv(args, pretrain_word_embedding, len(label2index))
            else:
                model = Bilstmcrf_cnn(args, pretrain_word_embedding, len(label2index))
        elif args.model_classes == 'cnn':
            model = CNN_BASE(args, pretrain_word_embedding, len(label2index),args.cnn_mode)
        elif args.model_classes in ['bilstm_cnn','bilstm_cnn_pool']:
            model = Bilstmcrf_cnn(args, pretrain_word_embedding, len(label2index))
        if args.use_dataParallel:
            model = nn.DataParallel(model.cuda())

        model = model.to(device)

        print('===============================开始训练================================')
        dev_result, dev_result_instance, lr, train_loss_step, train_loss_epoch, dev_loss_epoch = train(model, train_dataloader, dev_dataloader, 
                                                    args, device, tb_writer, index2label, tag, train_logger, dev_data_raw)

        # Save and Result
        with codecs.open(result_dir + '/dev_result.txt', 'w', encoding='utf-8') as f:
            json.dump(dev_result, f, indent=4, ensure_ascii=False)
        with codecs.open(args.model_save_dir + '/learning_rate.txt', 'w', encoding='utf-8') as f:
            json.dump(lr, f, indent=4, ensure_ascii=False)
        with codecs.open(args.model_save_dir + '/train_loss_step.txt', 'w', encoding='utf-8') as f:
            json.dump(train_loss_step, f, indent=4, ensure_ascii=False)
        with codecs.open(args.model_save_dir + '/train_loss_epoch.txt', 'w', encoding='utf-8') as f:
            json.dump(train_loss_epoch, f, indent=4, ensure_ascii=False)
        with codecs.open(args.model_save_dir + '/dev_loss_epoch.txt', 'w', encoding='utf-8') as f:
            json.dump(dev_loss_epoch, f, indent=4, ensure_ascii=False)

        print(time.time() - start_time)

        opt = vars(args)  # dict
        # save config
        opt["time'min"] = (time.time() - start_time) / 60
        save_config(opt, args.model_save_dir + '/args_config.json', verbose=True)
        train_logger.info("Train Time cost:{}min".format((time.time() - start_time) / 60))

    if args.do_test:
        print('=========================测试集==========================')
        args.test = True
        # Dataset
        if args.use_multi_token_mtl < 0:
            test_data, test_mask, test_label = pregress(test_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
            test_data = torch.tensor([f for f in test_data], dtype=torch.long)
            test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
            test_label = torch.tensor([f for f in test_label], dtype=torch.long)
            test_dataset = TensorDataset(test_data, test_mask, test_label)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        elif args.use_multi_token_mtl >= 0:
            test_data, test_mask, test_label,test_token_id = pregress_mtl(test_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length,mode=args.use_multi_token_mtl)
            test_data = torch.tensor([f for f in test_data], dtype=torch.long)
            test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
            test_label = torch.tensor([f for f in test_label], dtype=torch.long)
            test_token = torch.tensor([f for f in test_token_id], dtype=torch.long)
            test_dataset = TensorDataset(test_data, test_mask, test_label,test_token)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        print(args)

        # Model
        if args.model_classes == 'bilstm':
            if args.use_adv:
                test_model = Bilstmcrf_adv(args, pretrain_word_embedding, len(label2index))
            else:
                test_model = Bilstmcrf_cnn(args, pretrain_word_embedding, len(label2index))
        elif args.model_classes == 'cnn':
            test_model = CNN_BASE(args, pretrain_word_embedding, len(label2index),args.cnn_mode)
        elif args.model_classes in ['bilstm_cnn','bilstm_cnn_pool']:
            test_model = Bilstmcrf_cnn(args, pretrain_word_embedding, len(label2index))

        if args.use_dataParallel:
            test_model = nn.DataParallel(test_model.cuda())
        test_model = test_model.to(device)

        # Save and Result
        entity_model_save_dir = args.model_save_dir + 'entity_best.pt'
        entity_metric, entity_metric_instance, y_pred_entity = load_predict(test_model, test_dataloader,
                                                                            entity_model_save_dir,
                                                                            train_logger, index2label, tag, args,
                                                                            device, test_data_raw)


        with codecs.open(result_dir + '/test_result_entitymodel.txt', 'w', encoding='utf-8') as f:
            json.dump(entity_metric, f, indent=4, ensure_ascii=False)

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
