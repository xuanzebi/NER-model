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

package_dir_b = "D:/Paper_Shiyan/NER-model"
sys.path.insert(0, package_dir_b)

import warnings

warnings.filterwarnings("ignore")
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from util.util import get_logger, compute_f1, compute_spans_bio, compute_spans_bieos, compute_instance_f1
from model.lstm.lstmcrf import Bilstmcrf
from model.helper.get_data import get_cyber_data, pregress
from model.bert.utils_ner import convert_examples_to_features, read_examples_from_file, get_labels


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


def evaluate(data, model, label_map, tag, args, train_logger, device, dev_test_data, mode, pad_token_label_id):
    print("Evaluating on {} set...".format(mode))
    test_iterator = tqdm(data, desc="dev_test_interation")
    preds = None
    out_label_ids = None
    test_loss = 0.
    nb_eval_steps = 0

    for step, test_batch in enumerate(test_iterator):
        print(len(test_batch))
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)
        with torch.no_grad():
            _batch = tuple(t.to(device) for t in test_batch)
            input_ids, input_mask, segment_ids, label_ids = _batch
            loss, logits = model(input_ids, input_mask, segment_ids, labels=label_ids)
        test_loss += loss.item()
        nb_eval_steps += 1

        if args.use_dataParallel:
            loss = torch.sum(loss)  # if DataParallel model.module
        if args.use_crf == False:
            logits = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)

        label_ids = label_ids.to('cpu').numpy()
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

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
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            model.zero_grad()
            _batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = _batch
            # loss, _ = model.module.calculate_loss(input_ids, input_mask, label_ids)
            loss, _ = model(input_ids, input_mask, segment_ids, labels=label_ids)

            if args.use_dataParallel:
                loss = torch.sum(loss)  # if DataParallel

            tr_loss += loss.item()
            avg_loss += loss.item() / len(train_dataloader)
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(scheduler.get_lr()[0])
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                lr[epoch].append(scheduler.get_lr()[0])
                logging_loss = tr_loss

            epoch_iterator.set_postfix(train_loss=loss.item())

        tq.set_postfix(avg_loss=avg_loss)

        print('%d epoch，global_step: %d ,train_loss: %.2f' % (epoch, global_step, tr_loss / global_step))
        train_logger.info('%d epoch，global_step: %d ,train_loss: %.2f' % (epoch, global_step, tr_loss / global_step))

        metric, metric_instance = evaluate(dev_dataloader, model, label_map, tag, args, train_logger, device,
                                           dev_test_data, 'dev', pad_token_label_id)
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


def load_predict(model, data, model_save_dir, logger, label_map, tag, args, device, test_data, pad_token_label_id):
    start_time = time.time()
    model.load_state_dict(torch.load(model_save_dir))
    metric, metric_instance, y_pred = evaluate(data, model, label_map, tag, args, logger, device, test_data, 'test',
                                               pad_token_label_id)
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


def load_and_cache_examples(data, args, tokenizer, labels, pad_token_label_id, mode, logger):
    logger.info("Creating features from dataset file at %s", args.data_path)
    examples = read_examples_from_file(data, mode)
    features = convert_examples_to_features(
        examples,
        labels,
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
    parser.add_argument('--model_save_dir', type=str, default='D:/Paper_Shiyan/NER-model/save_models/test',
                        help='Root dir for saving models.')
    parser.add_argument('--tensorboard_dir', default='D:/Paper_Shiyan/NER-model/save_models/test/runs/', type=str)
    parser.add_argument('--data_path', default='D:/Paper_Shiyan/NER-model/data/ResumeNER/json_data', type=str,
                        help='数据路径')
    parser.add_argument('--pred_embed_path', default='', type=str,
                        help="预训练词向量路径,'cc.zh.300.vec','sgns.baidubaike.bigram-char','Tencent_AILab_ChineseEmbedding.txt'")
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
    parser.add_argument('--deal_long_short_data', default='cut', choices=['cut', 'pad', 'stay'], type=str,
                        help='对长文本或者短文本在验证测试的时候如何处理')
    parser.add_argument('--save_embed_path',
                        default='', type=str,
                        help='词向量存储路径')
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
                        )
    parser.add_argument("--model_name_or_path", default='D:/projects/nlp/bert/chinese_12_768_pytorch', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")

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

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
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
    parser.add_argument("--warmup_proportion", default=0.1, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some. 0/0.01")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

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

    labels = get_labels(new_data)
    index2label = {i: label for i, label in enumerate(labels)}
    label2index = {j: i for i, j in index2label.items()}
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # MODEL
    # args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(labels))
    model = BertForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
    # model = nn.DataParallel(model.cuda())
    model = model.to(device)
    param_optimizer = list(model.named_parameters())

    # DATASET
    train_dataset = load_and_cache_examples(train_data_raw, args, tokenizer, labels, pad_token_label_id, 'train',
                                            train_logger)
    test_dataset = load_and_cache_examples(test_data_raw, args, tokenizer, labels, pad_token_label_id, 'test',
                                           train_logger)
    dev_dataset = load_and_cache_examples(dev_data_raw, args, tokenizer, labels, pad_token_label_id, 'dev',
                                          train_logger)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_logger.info("Let's use{}GPUS".format(torch.cuda.device_count()))

    tb_writer = SummaryWriter(args.tensorboard_dir)

    if args.do_train:
        print('===============================开始训练================================')
        dev_result, dev_result_instance, lr = train(model, train_dataloader, dev_dataloader, args, device, tb_writer, \
                                                    index2label, tag, train_logger, dev_data_raw, pad_token_label_id)

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

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(labels))
        test_model = BertForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
        # model = nn.DataParallel(model.cuda())
        test_model = test_model.to(device)

        # token_model_save_dir = args.model_save_dir + 'token_best.pt'
        entity_model_save_dir = args.model_save_dir + 'entity_best.pt'

        # token_metric,token_metric_instance,y_pred_token = load_predict(test_model,test_dataloader,token_model_save_dir,train_logger,index2label,tag,args,device)
        entity_metric, entity_metric_instance, y_pred_entity = load_predict(test_model, test_dataloader,
                                                                            entity_model_save_dir,
                                                                            train_logger, index2label, tag, args,
                                                                            device, test_data_raw, pad_token_label_id)

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
