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
from model.lstm.lstmcrf import Bilstmcrf,FGM
from model.lstm.model import Bilstm_CRF_MTL,Bilstm_ST_END,Bilstm_MTL
from model.helper.get_data import get_cyber_data, pregress,pregress_mtl
from model.helper.convert_data import gen_token_ner_lstm, cys_label
from model.helper.evaluate import evaluate_crf,evaluate_instance,evaluate_st_end
from model.lstm.helper import evaluate,train_mtl
from model.helper.batchsamper import MultitastdataBatchsampler

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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

    if args.use_fgm:
        fgm = FGM(model)

    for epoch in tq:
        avg_loss = 0.
        epoch_start_time = time.time()
        model.train()
        model.zero_grad()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.zero_grad()
            _batch = tuple(t.to(device) for t in batch)
            
            if args.model_classes == 'bilstm':
                input_ids, input_mask, label_ids = _batch 
                if args.use_fgm:
                    loss, _ = model(input_ids, input_mask, label_ids)
                    loss.backward()
                    fgm.attack()
                    loss_adv,_ = model(input_ids, input_mask, label_ids)
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore() # 恢复embedding参数
                    # 梯度下降，更新参数
                    optimizer.step()
                    model.zero_grad()
                else:
                    if args.use_packpad:
                        loss, _ = model.forward_pack(input_ids, input_mask, label_ids)
                    else:
                        loss, _ = model(input_ids, input_mask, label_ids)
                    loss.backward()
                    optimizer.step()
            elif args.model_classes == 'bilstm_mtl':
                input_ids, input_mask, label_ids,token_id = _batch 
                loss, _ = model(input_ids, input_mask, label_ids,token_id)
                loss.backward()
                optimizer.step()
            elif args.model_classes == 'bilstm_data_mtl':
                input_ids, input_mask, label_ids,token_id,data_type = _batch 
                a = [int(k) for k in data_type]
                a = set(a)
                assert len(a) == 1  # 确保每个batch里只有一个data_type
                data_type_id = int(data_type[0])
                loss,_=model(input_ids, input_mask, label_ids,token_id,data_type_id)
                loss.backward()
                optimizer.step()
            elif args.model_classes == 'bilstm_start_end':
                input_ids, input_mask, start_id,end_id = _batch  
                loss, _, _ = model(input_ids,input_mask,start_id,end_id)
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

        if args.model_classes == 'bilstm_start_end':
            metric, metric_instance = evaluate_st_end(dev_dataloader, model, args.label_entity, tag, args, train_logger, device,dev_test_data, 'dev','lstm')
        else:
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

        # releax-f1 token-level f1
        if metric_instance['micro-f1'] > bestscore_instance:
            bestscore_instance = metric_instance['micro-f1']
            best_epoch_instance = epoch
            # print('token级别的F1best model epoch is: %d' % epoch)
            # train_logger.info('token级别的F1best model epoch is: %d' % epoch)
            # model_name = args.model_save_dir + "token_best.pt"
            # torch.save(model.state_dict(), model_name)

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
        test_result_instance.append(metric_instance)

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

def load_tensor_data_mtl(data,word2index,label2index,args,data_type):
    train_data_id, train_mask_id, train_label_id,train_token_id = pregress_mtl(data, word2idx, label2index,max_seq_lenth=args.max_seq_length)
    train_data = torch.tensor([f for f in train_data_id], dtype=torch.long)
    train_mask = torch.tensor([f for f in train_mask_id], dtype=torch.long)
    train_label = torch.tensor([f for f in train_label_id], dtype=torch.long)
    train_token = torch.tensor([f for f in train_token_id], dtype=torch.long)
    if data_type == 'cyber':
        data_type_tensor = torch.tensor([1]*train_data.size(0),dtype=torch.long)
    elif data_type == 'msra':
        data_type_tensor = torch.tensor([2]*train_data.size(0),dtype=torch.long)
    return train_data,train_mask,train_label,train_token,data_type_tensor

def mix_data_tensor(origin_data,new_data):
    origin_data, origin_mask,origin_label,origin_token,origin_data_type = origin_data
    new_data, new_mask,new_label,new_token,new_data_type = new_data
    train_data = torch.cat((origin_data,new_data),0)
    train_mask = torch.cat((origin_mask,new_mask),0)
    train_label = torch.cat((origin_label,new_label),0)
    train_token = torch.cat((origin_token,new_token),0)
    train_data_type = torch.cat((origin_data_type,new_data_type),0)
    train_dataset = TensorDataset(train_data,train_mask,train_label,train_token,train_data_type)
    return train_dataset

if __name__ == "__main__":
    print(os.getcwd())
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_test", default=False, type=str2bool, help="Whether to run test on the test set.")
    parser.add_argument('--model_classes', type=str, default='bilstm_data_mtl',choices=['bilstm','bilstm_mtl','bilstm_start_end',"bilstm_data_mtl"], help='Which model to choose.')
    parser.add_argument('--model_save_dir', type=str, default='/opt/hyp/NER/NER-model/saved_models/test/',
                        help='Root dir for saving models.')
    parser.add_argument('--tensorboard_dir', default='/opt/hyp/NER/NER-model/saved_models/test/runs/', type=str)
    parser.add_argument('--data_path', default='/opt/hyp/NER/NER-model/data/json_data', type=str,help='数据路径')
    parser.add_argument('--data_mtl_path', default='/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data', type=str,help='多任务学习的数据路径')
    parser.add_argument('--pred_embed_path', default='/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt', type=str,
                        help="预训练词向量路径,'cc.zh.300.vec','sgns.baidubaike.bigram-char','Tencent_AILab_ChineseEmbedding.txt'")
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
    parser.add_argument('--deal_long_short_data', default='cut', choices=['cut', 'pad', 'stay'], type=str,
                        help='对长文本或者短文本在验证测试的时候如何处理')
    parser.add_argument('--save_embed_path',
                        default='/opt/hyp/NER/NER-model/data/embedding/Tencent_AILab_ChineseEmbedding_cyber.p', type=str,
                        help='词向量存储路径')

    # parser.add_argument('--data_type', default='conll', help='数据类型 -conll - cyber')
    parser.add_argument("--use_bieos", default=True, type=str2bool, help="True:BIEOS False:BIO")
    parser.add_argument('--save_best_model', type=str2bool, default=False, help='Whether to save best model.')
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
    parser.add_argument('--use_packpad', default=False, type=str2bool, help='是否使用packed_pad')
    parser.add_argument('--use_fgm', default=False, type=str2bool, help='是否使用对抗样本')
    parser.add_argument('--use_token_mtl', default=False, type=str2bool, help='是否使用token级别多任务')
    parser.add_argument('--data_mtl', default=False, type=str2bool, help='是否使用数据级别多任务')

    parser.add_argument("--learning_rate", default=0.015, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_seq_length', default=200, type=int, help='Sequence max_length.')
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--word_emb_dim', default=200, type=int, help='预训练词向量的维度')
    parser.add_argument('--char_emb_dim', default=30, type=int)
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

    if args.model_classes == 'bilstm_data_mtl':

        args.save_embed_path = '/opt/hyp/NER/NER-model/data/embedding/Tencent_AILab_ChineseEmbedding_msra.p'

        msra_train_data_raw = json.load(open(args.data_mtl_path + '/train_data.json', encoding='utf-8'))
        msra_test_data_raw = json.load(open(args.data_mtl_path + '/test_data.json', encoding='utf-8'))
        msra_dev_data_raw = json.load(open(args.data_mtl_path + '/dev_data.json', encoding='utf-8'))
    
        msra_new_data = []
        msra_new_data.extend(msra_train_data_raw)
        msra_new_data.extend(msra_test_data_raw)
        msra_new_data.extend(msra_dev_data_raw)

        msra_pretrain_word_embedding, msra_vocab, msra_word2idx, msra_idx2word, msra_label2index, msra_index2label = get_cyber_data(msra_new_data, args)
        args.label_msra = msra_label2index
        train_logger.info('多任务学习的训练集大小为{}'.format(len(msra_new_data)))

    args.label = label2index
    args.vocab_size = len(vocab)
    args.label_entity = cys_label

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_logger.info("Let's use{}GPUS".format(torch.cuda.device_count()))

    tb_writer = SummaryWriter(args.tensorboard_dir)

    if args.do_train:
        # Dataset
        if args.model_classes == 'bilstm':
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
        elif args.model_classes == 'bilstm_mtl':
            train_data_id, train_mask_id, train_label_id,train_token_id = pregress_mtl(train_data_raw, word2idx, label2index,max_seq_lenth=args.max_seq_length)
            train_data = torch.tensor([f for f in train_data_id], dtype=torch.long)
            train_mask = torch.tensor([f for f in train_mask_id], dtype=torch.long)
            train_label = torch.tensor([f for f in train_label_id], dtype=torch.long)
            train_token = torch.tensor([f for f in train_token_id], dtype=torch.long)
            
            train_dataset = TensorDataset(train_data, train_mask, train_label,train_token)

            dev_data, dev_mask, dev_label,dev_token_id = pregress_mtl(dev_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
            dev_data = torch.tensor([f for f in dev_data], dtype=torch.long)
            dev_mask = torch.tensor([f for f in dev_mask], dtype=torch.long)
            dev_label = torch.tensor([f for f in dev_label], dtype=torch.long)
            dev_token = torch.tensor([f for f in dev_token_id], dtype=torch.long)
            dev_dataset = TensorDataset(dev_data, dev_mask, dev_label,dev_token)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
        elif args.model_classes == 'bilstm_data_mtl':
            """   data_type{0:cyber,1:msra}          """
            cyber_dataset = load_tensor_data_mtl(train_data_raw,word2idx,label2index,args,'cyber')
            msra_dataset = load_tensor_data_mtl(msra_new_data,msra_word2idx,msra_label2index,args,'msra')
            train_dataset = mix_data_tensor(cyber_dataset,msra_dataset)
            
            dev_data,dev_mask,dev_label,dev_token,dev_data_type = load_tensor_data_mtl(dev_data_raw,word2idx,label2index,args,'cyber')
            dev_dataset = TensorDataset(dev_data,dev_mask,dev_label,dev_token,dev_data_type)

            train_sampler = RandomSampler(train_dataset)
            multitasksamper = MultitastdataBatchsampler(train_sampler, batch_size=args.batch_size, drop_last=False, data_type=[1, 2])
            train_dataloader = DataLoader(train_dataset, batch_sampler=multitasksamper)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
        elif args.model_classes == "bilstm_start_end":
            train_data_id, train_mask_id, start_id, end_id= gen_token_ner_lstm(train_data_raw, args.max_seq_length, word2idx, cys_label)
            train_data = torch.tensor([f for f in train_data_id], dtype=torch.long)
            train_mask = torch.tensor([f for f in train_mask_id], dtype=torch.long)
            train_start = torch.tensor([f for f in start_id], dtype=torch.long)
            train_end = torch.tensor([f for f in end_id], dtype=torch.long)
            train_dataset = TensorDataset(train_data, train_mask, train_start,train_end)

            dev_data, dev_mask, dev_start_id,dev_end_id = gen_token_ner_lstm(dev_data_raw, args.max_seq_length, word2idx, cys_label)
            dev_data = torch.tensor([f for f in dev_data], dtype=torch.long)
            dev_mask = torch.tensor([f for f in dev_mask], dtype=torch.long)
            dev_start = torch.tensor([f for f in dev_start_id], dtype=torch.long)
            dev_end = torch.tensor([f for f in dev_end_id], dtype=torch.long)
            dev_dataset = TensorDataset(dev_data, dev_mask, dev_start,dev_end)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)


        # Model
        if args.model_classes == 'bilstm':
            model = Bilstmcrf(args, pretrain_word_embedding, len(label2index))
        elif args.model_classes == 'bilstm_mtl':
            print('===============================多任务================================')
            model = Bilstm_CRF_MTL(args,pretrain_word_embedding,len(label2index))
        elif args.model_classes == 'bilstm_data_mtl':
            print('===============================数据多任务================================')
            model = Bilstm_MTL(args,pretrain_word_embedding,msra_pretrain_word_embedding,len(label2index),len(msra_label2index))
        elif args.model_classes == "bilstm_start_end":
            print("=====================双指针======================")
            model = Bilstm_ST_END(args,pretrain_word_embedding,len(cys_label))

        if args.use_dataParallel:
            model = nn.DataParallel(model.cuda())
        model = model.to(device)

        print('===============================开始训练================================')
        dev_result, dev_result_instance, lr, train_loss_step, train_loss_epoch, dev_loss_epoch = train(model, train_dataloader, dev_dataloader, 
                                                    args, device, tb_writer, index2label, tag, train_logger, dev_data_raw)

        # Save and Result
        with codecs.open(result_dir + '/dev_result.txt', 'w', encoding='utf-8') as f:
            json.dump(dev_result, f, indent=4, ensure_ascii=False)

        # with codecs.open(result_dir + '/dev_result_instance.txt', 'w', encoding='utf-8') as f:
        #     json.dump(dev_result_instance, f, indent=4, ensure_ascii=False)

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
        # Dataset
        if args.model_classes == 'bilstm':
            test_data, test_mask, test_label = pregress(test_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
            test_data = torch.tensor([f for f in test_data], dtype=torch.long)
            test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
            test_label = torch.tensor([f for f in test_label], dtype=torch.long)
            test_dataset = TensorDataset(test_data, test_mask, test_label)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        elif args.model_classes == 'bilstm_mtl':
            test_data, test_mask, test_label,test_token_id = pregress_mtl(test_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
            test_data = torch.tensor([f for f in test_data], dtype=torch.long)
            test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
            test_label = torch.tensor([f for f in test_label], dtype=torch.long)
            test_token = torch.tensor([f for f in test_token_id], dtype=torch.long)
            test_dataset = TensorDataset(test_data, test_mask, test_label,test_token)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        elif args.model_classes == 'bilstm_data_mtl':
            test_data,test_mask,test_label,test_token,test_data_type = load_tensor_data_mtl(test_data_raw,word2idx,label2index,args,'cyber')
            test_dataset = TensorDataset(test_data,test_mask,test_label,test_token,test_data_type)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        elif args.model_classes == "bilstm_start_end":
            test_data_id, test_mask_id, test_start_id, test_end_id= gen_token_ner_lstm(test_data_raw, args.max_seq_length, word2idx, cys_label)
            test_data = torch.tensor([f for f in test_data_id], dtype=torch.long)
            test_mask = torch.tensor([f for f in traintest_mask_id_mask_id], dtype=torch.long)
            test_start = torch.tensor([f for f in test_start_id], dtype=torch.long)
            test_end = torch.tensor([f for f in test_end_id], dtype=torch.long)
            test_dataset = TensorDataset(test_data, test_mask, test_start,test_end)

        print(args)

        # Model
        if args.model_classes == 'bilstm':
            test_model = Bilstmcrf(args, pretrain_word_embedding, len(label2index))
        elif args.model_classes == 'bilstm_mtl':
            test_model = Bilstm_CRF_MTL(args,pretrain_word_embedding,len(label2index))
        elif args.model_classes == 'bilstm_data_mtl':
            test_model = Bilstm_MTL(args,pretrain_word_embedding,msra_pretrain_word_embedding,len(label2index),len(msra_label2index))
        elif args.model_classes == "bilstm_start_end":
            test_model = Bilstm_ST_END(args,pretrain_word_embedding,len(cys_label))

        if args.use_dataParallel:
            test_model = nn.DataParallel(test_model.cuda())
        test_model = test_model.to(device)

        # Save and Result
        # token_model_save_dir = args.model_save_dir + 'token_best.pt'
        entity_model_save_dir = args.model_save_dir + 'entity_best.pt'

        # token_metric,token_metric_instance,y_pred_token = load_predict(test_model,test_dataloader,token_model_save_dir,train_logger,index2label,tag,args,device)
        entity_metric, entity_metric_instance, y_pred_entity = load_predict(test_model, test_dataloader,
                                                                            entity_model_save_dir,
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
