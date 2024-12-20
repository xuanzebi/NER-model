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
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import codecs
import json
from sklearn.model_selection import StratifiedKFold, KFold

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

import sys

package_dir_b = "/opt/hyp/NER/NER-model"
sys.path.insert(0, package_dir_b)

import warnings

warnings.filterwarnings("ignore")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from util.util import get_logger, compute_f1, compute_spans_bio, compute_spans_bieos, compute_instance_f1
from model.helper.get_data import get_cyber_data, pregress
from model.helper.query_map import query_sign_map
from model.bert.utils_ner import convert_examples_to_features, read_examples_from_file, get_labels
from model.bert.util_data_helper import Doubue_convert_examples_to_features,cys_label,msra_label,gen_query_ner,MRC_convert_examples_to_features
from model.bert.bert_model import BertForDoublePointClassification,BertforMrcNER
from model.helper.evaluate import evaluate_crf,evaluate_instance,evaluate_st_end,evaluate_mrc_ner
from model.lstm.lstmcrf import FGM

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def evaluate(data, model, label_map, tag, args, train_logger, device, dev_test_data, mode, pad_token_label_id):
    print("Evaluating on {} set...".format(mode))
    test_iterator = tqdm(data, desc="dev_test_interation")
    preds = None
    out_label_ids = None
    test_loss = 0.
    nb_eval_steps = 0

    for step, test_batch in enumerate(test_iterator):
        # print(len(test_batch))
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)
        with torch.no_grad():
            _batch = tuple(t.to(device) for t in test_batch)
            input_ids, input_mask, segment_ids, label_ids = _batch
            loss, logits = model(input_ids, input_mask, segment_ids, labels=label_ids)
        nb_eval_steps += 1
        if args.use_dataParallel:
            loss = torch.sum(loss)  # if DataParallel model.module
        test_loss += loss.item()

        if args.use_crf == False:
            logits = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.cpu().numpy(), axis=0)

        test_iterator.set_postfix(test_loss=loss.item())

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    metric_instance = evaluate_instance(out_label_list, preds_list)
    metric = evaluate_crf(out_label_list, preds_list, tag)
    metric['test_loss'] = test_loss / nb_eval_steps

    if mode == 'test':
        return metric, metric_instance, preds_list
    else:
        return metric, metric_instance


def train(model, train_dataloader, dev_dataloader, args, device, tb_writer, label_map, tag, train_logger,
          dev_test_data, pad_token_label_id):
    # param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    train_loss_step = {}
    train_loss_epoch = {}
    dev_loss_epoch = {}
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    warmup_steps = int(args.warmup_proportion * t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    if args.use_fgm:
        fgm = FGM(model)

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
        epoch_start_time = time.time()
        avg_loss = 0.
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            model.zero_grad()
            _batch = tuple(t.to(device) for t in batch)
            if args.model_class == 'bert':
                input_ids, input_mask, segment_ids, label_ids = _batch
                if args.use_fgm:
                    loss, _ = model(input_ids, input_mask, segment_ids, labels=label_ids)
                    loss.backward()
                    fgm.attack()
                    loss_adv,_ = model(input_ids, input_mask, segment_ids, labels=label_ids)
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore() # 恢复embedding参数
                else:
                    loss, _ = model(input_ids, input_mask, segment_ids, labels=label_ids)
            elif args.model_class == "bert_double":
                input_ids, input_mask, segment_ids, start_ids,end_ids = _batch
                loss,_,_ = model(input_ids, input_mask, segment_ids, start_ids=start_ids,end_ids=end_ids)
            elif args.model_class == "bert_mrc":
                input_ids, input_mask, segment_ids, start_ids,end_ids,ner_cate = _batch
                loss,_,_ = model(input_ids, input_mask, segment_ids, start_ids=start_ids,end_ids=end_ids)

            if args.use_dataParallel:
                loss = torch.sum(loss)  # if DataParallel

            tr_loss += loss.item()
            avg_loss += loss.item() / len(train_dataloader)

            if not args.use_fgm:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print('当前epoch:{}, step:{} 的学习率为:{}!'.format(epoch, step, scheduler.get_lr()[0]))
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                train_loss_step[global_step] = (tr_loss - logging_loss) / args.logging_steps
                lr[epoch].append(scheduler.get_lr()[0])
                logging_loss = tr_loss

            epoch_iterator.set_postfix(train_loss=loss.item())

        tq.set_postfix(avg_loss=avg_loss)

        train_loss_epoch[epoch] = avg_loss
        if args.model_class == 'bert':
            metric, metric_instance = evaluate(dev_dataloader, model, label_map, tag, args, train_logger, device,
                                            dev_test_data, 'dev', pad_token_label_id)
        elif args.model_class == 'bert_double':
            metric, metric_instance = evaluate_st_end(dev_dataloader, model, args.label_entity, tag, args, train_logger, device,
                                            dev_test_data, 'dev',pad_token_label_id,'bert')
        elif args.model_class == 'bert_mrc':
            metric, metric_instance = evaluate_mrc_ner(dev_dataloader, model, args.label_entity, tag, args, train_logger, device,
                                            dev_test_data, 'dev',pad_token_label_id,'bert')
            
        metric_instance['epoch'] = epoch
        metric['epoch'] = epoch
        dev_loss_epoch[epoch] = metric['test_loss']
        # print(metric['test_loss'], epoch)
        # train_logger.info("epoch{},test_loss{}".format(metric['test_loss'], epoch))

        tb_writer.add_scalar('test_loss', metric['test_loss'], epoch)

        if metric['micro-f1'] > bestscore:
            bestscore = metric['micro-f1']
            best_epoch = epoch
            print('实体级别的F1的best model epoch is: %d' % epoch)
            train_logger.info('实体级别的F1的best model epoch is: %d' % epoch)
            model_name = args.model_save_dir + "pytorch_model.bin"
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), model_name)
            # model_to_save.save_pretrained(args.model_save_dir)
            # tokenizer.save_pretrained(args.model_save_dir)

        # releax-f1 token-level f1
        if metric_instance['micro-f1'] > bestscore_instance:
            bestscore_instance = metric_instance['micro-f1']
            best_epoch_instance = epoch
            # print('token级别的F1best model epoch is: %d' % epoch)
            # train_logger.info('token级别的F1best model epoch is: %d' % epoch)
            # model_name = args.model_save_dir + "token_best.pt"
            # torch.save(model.state_dict(), model_name)

        
        print('epoch:{} , global_step:{}, train_loss:{}, train_avg_loss:{}, dev_avg_loss:{},epoch耗时:{}!'.format(epoch, global_step, 
                                                    tr_loss / global_step,avg_loss, metric['test_loss'],time.time()-epoch_start_time))
        train_logger.info('epoch:{}, global_step:{}, train_loss:{},train_avg_loss:{},dev_avg_loss:{},epoch耗时:{}!'.format(epoch, global_step, 
                                                     tr_loss / global_step,avg_loss,metric['test_loss'],time.time()-epoch_start_time))
        print('epoch:{} P:{}, R:{}, F1:{} ,best F1:{}!"\n"'.format(epoch, metric['precision-overall'],metric['recall-overall'],
                                                                         metric['f1-measure-overall'], bestscore))
        train_logger.info('epoch:{} P:{}, R:{}, F1:{},best F1:{}! "\n"'.format(epoch, metric['precision-overall'], metric['recall-overall'],
                                                                     metric['f1-measure-overall'], bestscore))

        test_result.append(metric)
        test_result_instance.append(metric_instance)

    test_result.append({'best_dev_f1': bestscore,
                        'best_dev_epoch': best_epoch})
    test_result_instance.append({'best_dev_f1': bestscore_instance,
                                 'best_dev_epoch': best_epoch_instance})
    tb_writer.close()
    return test_result, test_result_instance, lr, train_loss_step, train_loss_epoch,dev_loss_epoch



def save_config(config, path, verbose=True):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config


def load_and_cache_examples(data, args, tokenizer, label2index, pad_token_label_id, mode, logger):
    logger.info("Creating features from dataset file at %s", args.data_path)
    examples = read_examples_from_file(data, mode)
    features = convert_examples_to_features(
        examples,
        label2index,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=pad_token_label_id,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def load_double_examples(data, args, tokenizer, pad_token_label_id, mode, logger):
    logger.info("Creating features from dataset file at %s", args.data_path)
    examples = read_examples_from_file(data, mode)
    features = Doubue_convert_examples_to_features(
        examples,
        args.label_entity,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=pad_token_label_id,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids)
    return dataset

def load_mrc_examples(data, args, tokenizer, pad_token_label_id, mode, logger,allow_impossible):
    logger.info("Creating features from dataset file at %s", args.data_path)
    examples = gen_query_ner(data, query_sign_map,args.data_type)
    features = MRC_convert_examples_to_features(
        examples,
        args.label_entity,
        args.max_seq_length,
        tokenizer,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_label_id=pad_token_label_id,
        allow_impossible=allow_impossible,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_position for f in features], dtype=torch.long)
    ner_cate = torch.tensor([f.ner_cate for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,ner_cate)
    if mode == 'test':
        return dataset,features
    else:
        return dataset

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}

# MODEL_NUM = {"0":}


if __name__ == "__main__":
    print(os.getcwd())
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_test", default=True, type=str2bool, help="Whether to run test on the test set.")
    parser.add_argument('--save_best_model', type=str2bool, default=True, help='Whether to save best model.')
    parser.add_argument('--model_save_dir', type=str, default='/opt/hyp/NER/NER-model/saved_models/msra/msra_bert_mrc/',
                        help='Root dir for saving models.')
    parser.add_argument('--tensorboard_dir', default='/opt/hyp/NER/NER-model/saved_models/msra/msra_bert_mrc/runs/', type=str)
    parser.add_argument('--data_path', default='/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data', type=str,
                        help='数据路径')
    parser.add_argument('--pred_embed_path', default='', type=str,
                        help="预训练词向量路径,'cc.zh.300.vec','sgns.baidubaike.bigram-char','Tencent_AILab_ChineseEmbedding.txt'")
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
    parser.add_argument('--deal_long_short_data', default='cut', choices=['cut', 'pad', 'stay'], type=str,
                        help='对长文本或者短文本在验证测试的时候如何处理')
    parser.add_argument('--save_embed_path', default='', type=str, help='词向量存储路径')
    parser.add_argument("--model_type", default='bert', type=str, help="Model type selected in the list")
    parser.add_argument("--model_class", default='bert_mrc', type=str, choices=["bert","bert_double","bert_mrc"])
    parser.add_argument("--data_type", default='zh_msra_ner', type=str, choices=["cyber_sec_ch_ner","zh_msra_ner",""])
    parser.add_argument("--model_name_or_path", default='/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12_pytorch', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")

    # parser.add_argument('--data_type', default='conll', help='数据类型 -conll - cyber')
    parser.add_argument("--use_bieos", default=True, type=str2bool, help="True:BIEOS False:BIO")
    parser.add_argument('--token_level_f1', default=False, type=str2bool, help='Sequence max_length.')
    parser.add_argument('--do_lower_case', default=False, type=str2bool, help='False 不计算token-level f1，true 计算')
    parser.add_argument('--freeze', default=True, type=str2bool, help='是否冻结词向量')
    parser.add_argument('--use_crf', default=False, type=str2bool, help='是否使用crf')
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
    parser.add_argument('--use_fgm', default=False, type=str2bool, help='是否加上对抗信息扰动')

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_length', default=200, type=int, help='Sequence max_length.')
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--word_emb_dim', default=300, type=int, help='预训练词向量的维度')
    parser.add_argument('--char_emb_dim', default=30, type=int)
    parser.add_argument('--rnn_hidden_dim', default=128, type=int, help='rnn的隐状态的大小')
    parser.add_argument('--num_layers', default=1, type=int, help='rnn中的层数')
    parser.add_argument('--lr_decay', default=0.05, type=float)
    parser.add_argument('--momentum', default=0, type=float, help="0 or 0.9")
    parser.add_argument('--min_count', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float, help='词向量后的dropout')
    parser.add_argument('--dropoutlstm', default=0.5, type=float, help='lstm后的dropout')
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some. 0/0.01")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

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

    # label 确保每次顺序一样
    label2index = get_labels(new_data)
    print('该数据集的label为:',label2index)
    index2label = {j: i for i, j in label2index.items()}
    args.label = label2index

    if 'msra' in args.data_type:
        args.label_entity = msra_label
    if "cyber" in args.data_type:
        args.label_entity = cys_label

    pad_token_label_id = CrossEntropyLoss().ignore_index

    # MODEL
    # args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.model_class == 'bert':
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(label2index))
    elif args.model_class == 'bert_double':
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(args.label_entity))
    elif args.model_class == 'bert_mrc':
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=2)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_logger.info("Let's use{}GPUS".format(torch.cuda.device_count()))

    tb_writer = SummaryWriter(args.tensorboard_dir)

    if args.do_train:
        print('===============================开始训练================================')
        # Model 
        if args.model_class == 'bert':
            model = BertForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
        elif args.model_class == 'bert_double':
            model = BertForDoublePointClassification(args,config)
        elif args.model_class == 'bert_mrc':
            model = BertforMrcNER(args,config)

        if args.use_dataParallel:
            model = nn.DataParallel(model.cuda())
        model = model.to(device)
        param_optimizer = list(model.named_parameters())

        # DATASET
        if args.model_class == 'bert_double':
            train_dataset = load_double_examples(train_data_raw, args, tokenizer, pad_token_label_id, 'train', train_logger)
            dev_dataset = load_double_examples(dev_data_raw, args, tokenizer, pad_token_label_id, 'dev',train_logger)
        elif args.model_class == 'bert':    
            train_dataset = load_and_cache_examples(train_data_raw, args, tokenizer, label2index, pad_token_label_id, 'train', train_logger)
            dev_dataset = load_and_cache_examples(dev_data_raw, args, tokenizer, label2index, pad_token_label_id, 'dev',train_logger)
        elif args.model_class == 'bert_mrc':
            train_dataset = load_mrc_examples(train_data_raw, args, tokenizer, pad_token_label_id, 'train', train_logger,allow_impossible=True)
            dev_dataset = load_mrc_examples(dev_data_raw, args, tokenizer, pad_token_label_id, 'dev',train_logger,allow_impossible=False)   
       
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

        dev_result, dev_result_instance, lr, train_loss_step, train_loss_epoch,dev_loss_epoch = train(model, train_dataloader, dev_dataloader, args, device, tb_writer, \
                                                    index2label, tag, train_logger, dev_data_raw, pad_token_label_id)

        # Result and save
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

        del(model)

    if args.do_test:
        print('=========================测试集==========================')
        print(args)
        start_time = time.time()

        # TestData
        if args.model_class == 'bert':
            test_dataset = load_and_cache_examples(test_data_raw, args, tokenizer, label2index, pad_token_label_id, 'test',train_logger)
        elif args.model_class == 'bert_double':
            test_dataset = load_double_examples(test_data_raw, args, tokenizer, pad_token_label_id, 'test', train_logger)
        elif args.model_class == 'bert_mrc':
            test_dataset,test_features = load_mrc_examples(test_data_raw, args, tokenizer, pad_token_label_id, 'test',train_logger,allow_impossible=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        # Model
        entity_model_save_dir = args.model_save_dir + 'pytorch_model.bin'
        if args.model_class == 'bert':
            model = BertForTokenClassification.from_pretrained(entity_model_save_dir,config=config)
            model = model.to(device)
            # 如果命名不是pytorch_model.bin的话，需要load_state_dict
            model.load_state_dict(torch.load(entity_model_save_dir))
            for param in model.parameters():
                param.requires_grad = False
            if args.use_dataParallel:
                model = nn.DataParallel(model.cuda())
            # Result and save
            entity_metric, entity_metric_instance, y_pred_entity = evaluate(test_dataloader, model, index2label, tag, args, train_logger, device, test_data_raw, 'test',
                                                pad_token_label_id)
        elif args.model_class == 'bert_double':
            model = BertForDoublePointClassification(args,config)
            model = model.to(device)
            # 如果命名不是pytorch_model.bin的话，需要load_state_dict
            model.load_state_dict(torch.load(entity_model_save_dir))
            for param in model.parameters():
                param.requires_grad = False
            if args.use_dataParallel:
                model = nn.DataParallel(model.cuda())
            entity_metric, entity_metric_instance,y_pred_entity = evaluate_st_end(test_dataloader, model, args.label_entity, tag, args, train_logger, device,
                                            test_data_raw, 'test',pad_token_label_id,'bert')
        elif args.model_class == 'bert_mrc':
            model = BertforMrcNER(args,config)
            model = model.to(device)
            model.load_state_dict(torch.load(entity_model_save_dir))
            for param in model.parameters():
                param.requires_grad = False
            if args.use_dataParallel:
                model = nn.DataParallel(model.cuda())
            entity_metric, entity_metric_instance,y_pred_entity = evaluate_mrc_ner(test_dataloader, model, args.label_entity, tag, args, train_logger, device,
                                            test_data_raw, 'test',pad_token_label_id,'bert')

        end_time = time.time()
        print('预测Time Cost:{}s'.format(end_time - start_time))
        train_logger.info('预测Time Cost:{}s'.format(end_time - start_time))

        # token_model_save_dir = args.model_save_dir + 'token_best.pt'
        # token_metric,token_metric_instance,y_pred_token = load_predict(test_model,test_dataloader,token_model_save_dir,train_logger,index2label,tag,args,device)

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
        if args.model_class == 'bert':
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
        elif args.model_class == 'bert_mrc':
            pass