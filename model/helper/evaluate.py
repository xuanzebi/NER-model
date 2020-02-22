from tqdm import tqdm
import torch
import torch.nn.functional as F
from util.util import compute_f1, compute_spans_bio, compute_spans_bieos, compute_instance_f1

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

def get_tags_bert(start,end,label_map,input_mask):
    tags = []
    label_map = {int(i):j for i,j in label_map.items()}
    for i, st in enumerate(start):
        len_tag = sum(input_mask[i]==1) -2
        tag = ['O'] * len_tag  # CLS 和 SEP
        for j in range(1,len_tag+1):
            if input_mask[i][j] == 0:
                break
            if st[j] == 0:
                continue
            for k in range(j,len_tag+1):
                if end[i][k] == st[j]:
                    if k == j:
                        tag[k-1]= 'S-' +label_map[st[j]]
                    else:
                        tag[j-1] = 'B-' +label_map[st[j]]
                        for p in range(j+1,k):
                            tag[p-1] = 'I-' +label_map[st[j]]
                        tag[k-1] = 'E-' +label_map[st[j]]
                    break
                # 可选
                if end[i][k] != 0:
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
            for k in range(j,len(st)):
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
                # 可选
                # if end[i][k] != 0:
                #     break 
        tags.append(tag)
    return tags
                    

# 针对bilstm 的 双指针的召回label
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
            y_true = get_tags_bert(start_id,end_id,label_map,input_mask)
            y_pred = get_tags_bert(start_logits,end_logits,label_map,input_mask)

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