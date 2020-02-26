from tqdm import tqdm
import torch
import torch.nn.functional as F
from util.util import compute_f1, compute_spans_bio, compute_spans_bieos, compute_instance_f1
from model.helper.query_map import query_sign_map

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

def get_tags_bert(start,end,label_map,input_mask,args):
    """多个start指针，一个end如何处理，1个start，多个end如何处理"""
    tags = []
    label_map = {int(i):j for i,j in label_map.items()}
    for i, st in enumerate(start):
        len_tag = sum(input_mask[i]==1) -2
        tag = ['O'] * len_tag  # CLS 和 SEP
        for j in range(1,len_tag+1):
            if st[j] == 0:
                continue
            end_start_len = min(len_tag+1,j+30) # 30 为实体的长度
            for k in range(j,end_start_len):
                if end[i][k] == st[j]:
                    if k == j:
                        tag[k-1]= 'S-' +label_map[st[j]]
                    else:
                        tag[j-1] = 'B-' +label_map[st[j]]
                        for p in range(j+1,k):
                            if 'msra' in args.data_type:
                                tag[p-1] = 'M-' +label_map[st[j]]
                            else:
                                tag[p-1] = 'I-' +label_map[st[j]]
                        tag[k-1] = 'E-' +label_map[st[j]]
                    break
        tags.append(tag)
    return tags

def get_tags(start,end,label_map,input_mask):
    tags = []
    label_map = {int(i):j for i,j in label_map.items()}
    for i, st in enumerate(start):
        tag = ['O'] * sum(input_mask[i]==1)
        for j, m in enumerate(st):
            if input_mask[i][j] == 0:
                break
            if m == 0:
                continue
            entity_len = min(len(st),j+30)
            for k in range(j,entity_len):
                if input_mask[i][k] == 0:
                    break
                if end[i][k] == m:
                    if k == j:
                        tag[k]= 'S-' +label_map[m]
                    else:
                        tag[j] = 'B-' +label_map[m]
                        for p in range(j+1,k):
                            tag[p] = 'I-' +label_map[m]
                        tag[k] = 'E-' +label_map[m]
                    break
        tags.append(tag)
    return tags
                    

# 针对双指针的召回评估函数
def evaluate_st_end(data, model, label_map, tag, args, train_logger, device, dev_test_data, mode,pad_token_label_id=-100, model_name=None):
    print("Evaluating on {} set...".format(mode))
    test_iterator = tqdm(data, desc="dev_test_interation")
    Y_PRED = []
    Y_TRUE = []
    test_loss = 0.
    test_step = 0

    for step, test_batch in enumerate(test_iterator):
        # print(len(test_batch))
        test_step += 1
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)

        with torch.no_grad():
            if model_name == 'bert':
                input_ids, input_mask, segment_ids, start_id,end_id = _test_batch
                loss,start_logits,end_logits = model(input_ids, input_mask, segment_ids, start_ids=start_id,end_ids=end_id)
            elif model_name == 'lstm':
                input_ids, input_mask, start_id,end_id = _test_batch
                loss, start_logits,end_logits = model(input_ids, input_mask, start_id,end_id)

        if args.use_dataParallel:
            loss = torch.sum(loss)  # if DataParallel model.module

        if args.use_crf == False:
            start_logits = torch.argmax(F.log_softmax(start_logits, dim=-1), dim=-1)
            end_logits = torch.argmax(F.log_softmax(end_logits, dim=-1), dim=-1)

        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()

        test_loss += loss.item()
        start_id = start_id.to('cpu').numpy()
        end_id = end_id.to('cpu').numpy()
        input_mask = input_mask.cpu().data.numpy()

        if model_name == 'lstm':
            y_true = get_tags(start_id,end_id,label_map,input_mask)
            y_pred = get_tags(start_logits,end_logits,label_map,input_mask)
        elif model_name == 'bert':
            y_true = get_tags_bert(start_id,end_id,label_map,input_mask,args)
            y_pred = get_tags_bert(start_logits,end_logits,label_map,input_mask,args)

        Y_PRED.extend(y_pred)
        Y_TRUE.extend(y_true)

        test_iterator.set_postfix(test_loss=loss.item())


    metric_instance = evaluate_instance(Y_TRUE, Y_PRED)
    metric = evaluate_crf(Y_TRUE, Y_PRED, tag)
    metric['test_loss'] = test_loss / test_step

    if mode == 'test':
        return metric, metric_instance, Y_PRED
    else:
        return metric, metric_instance


def get_tags_mrc(args,start,end,label_map,input_mask,ner_cate):
    """多个start指针，一个end如何处理，1个start，多个end如何处理
        该函数是根据start指针来匹配，但是end指针只能用一次。"""
    tags = []
    label_map = {int(i):j for i,j in label_map.items()}
    query_info_dict = query_sign_map[args.data_type]['natural_query']
    for i, st in enumerate(start):
        cur_end = 0
        cur_cate = label_map[ner_cate[i]]
        len_query_cate = len(query_info_dict[cur_cate])
        len_text = sum(input_mask[i]==1)
        tag = ['O'] * (len_text - len_query_cate - 3)  # CLS 和 SEP
        for j in range(len_query_cate+2,len_text-1):
            if st[j] == 0:
                continue
            end_start_len = min(len_text-1,j+20) # 30 为实体的长度
            end_start = max(j,cur_end) # 多个start 1个end指针，end遍历用最近的
            for k in range(end_start,end_start_len):
                if end[i][k] == st[j]:
                    if k == j:
                        tag[k-len_query_cate-2] = 'S-' + cur_cate
                    else:
                        tag[j-len_query_cate-2] = 'B-' + cur_cate
                        for p in range(j+1,k):
                            if 'msra' in args.data_type:
                                tag[p-len_query_cate-2] = 'M-' + cur_cate
                            else:
                                tag[p-len_query_cate-2] = 'I-' + cur_cate
                        tag[k-len_query_cate-2] = 'E-' + cur_cate
                        cur_end = k + 1
                    break
                # 可选
                # if end[i][k] != 0:
                #     break 
        tags.append(tag)
    return tags


def get_tags_mrc_v2(args,start,end,label_map,input_mask,ner_cate):
    """多个start指针，一个end如何处理，1个start，多个end如何处理
        首先召回start指针里的第一个，如果在start和end中还有start则跳过"""
    tags = []
    label_map = {int(i):j for i,j in label_map.items()}
    query_info_dict = query_sign_map[args.data_type]['natural_query']
    for i, st in enumerate(start):
        cur_cate = label_map[ner_cate[i]]
        len_query_cate = len(query_info_dict[cur_cate])
        len_text = sum(input_mask[i]==1)
        tag = ['O'] * (len_text - len_query_cate - 3)  # CLS 和 SEP
        j = len_query_cate + 2
        while j < len_text-1:
            if st[j] == 0:
                j +=1
                continue
            end_start_len = min(len_text-1,j+20) # 30 为实体的长度
            for k in range(j,end_start_len):
                if end[i][k] == st[j]:
                    if k == j:
                        tag[k-len_query_cate-2] = 'S-' + cur_cate
                    else:
                        tag[j-len_query_cate-2] = 'B-' + cur_cate
                        for p in range(j+1,k):
                            if 'msra' in args.data_type:
                                tag[p-len_query_cate-2] = 'M-' + cur_cate
                            else:
                                tag[p-len_query_cate-2] = 'I-' + cur_cate
                        tag[k-len_query_cate-2] = 'E-' + cur_cate
                        j = k
                    break
            j += 1
                # 可选
                # if end[i][k] != 0:
                #     break 
        tags.append(tag)
    return tags

# 针对双指针的召回评估函数
def evaluate_mrc_ner(data, model, label_map, tag, args, train_logger, device, dev_test_data, mode,pad_token_label_id=-100, model_name=None):
    """ lalel_map: idx2label"""

    print("Evaluating on {} set...".format(mode))
    test_iterator = tqdm(data, desc="dev_test_interation")
    Y_PRED = []
    Y_TRUE = []
    test_loss = 0.
    test_step = 0

    for step, test_batch in enumerate(test_iterator):
        # print(len(test_batch))
        test_step += 1
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)

        with torch.no_grad():
            if model_name == 'bert':
                input_ids, input_mask, segment_ids, start_id,end_id,ner_cate = _test_batch
                loss,start_logits,end_logits = model(input_ids, input_mask, segment_ids, start_ids=start_id,end_ids=end_id)
            elif model_name == 'lstm':
                input_ids, input_mask, start_id,end_id = _test_batch
                loss, start_logits,end_logits = model(input_ids, input_mask, start_id,end_id)

        if args.use_dataParallel:
            loss = torch.sum(loss)  # if DataParallel model.module

        if args.use_crf == False:
            start_logits = torch.argmax(F.log_softmax(start_logits, dim=-1), dim=-1)
            end_logits = torch.argmax(F.log_softmax(end_logits, dim=-1), dim=-1)

        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()

        test_loss += loss.item()
        start_id = start_id.to('cpu').numpy()
        end_id = end_id.to('cpu').numpy()
        ner_cate = ner_cate.to('cpu').numpy()
        input_mask = input_mask.cpu().data.numpy()

        if model_name == 'lstm':
            y_true = get_tags(start_id,end_id,label_map,input_mask)
            y_pred = get_tags(start_logits,end_logits,label_map,input_mask)
            assert len(y_true) == len(y_pred)
        elif model_name == 'bert':
            y_true = get_tags_mrc_v2(args,start_id,end_id,label_map,input_mask,ner_cate)
            y_pred = get_tags_mrc_v2(args,start_logits,end_logits,label_map,input_mask,ner_cate)
            assert len(y_true) == len(y_pred)

        Y_PRED.extend(y_pred)
        Y_TRUE.extend(y_true)

        test_iterator.set_postfix(test_loss=loss.item())


    metric_instance = evaluate_instance(Y_TRUE, Y_PRED)
    metric = evaluate_crf(Y_TRUE, Y_PRED, tag)
    metric['test_loss'] = test_loss / test_step

    if mode == 'test':
        return metric, metric_instance, Y_PRED
    else:
        return metric, metric_instance