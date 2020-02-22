import json
from util.util import compute_spans_bieos

# 安全中文数据
cys_label = {"0": "O", "1": "RT", "2": "LOC", "3": 'PER', "4": "ORG", "5": "SW", "6": "VUL_ID"}


def gen_token_ner_lstm(dataset, max_seq_lenth, word2idx, index2label):
    label2index = {j: int(i) for i, j in index2label.items()}
    start_pos = []
    end_pos = []
    INPUT_ID = []
    INPUT_MASK = []
    for text, label in dataset:
       
        text = text.split(' ')
        label = label.split(' ')
        start_id = [0] * len(label)
        end_id = [0] * len(label)
        input_mask = [1] * len(text)
        candidate_span_label = compute_spans_bieos(label)
        if len(candidate_span_label) > 0:
            candidate_span_label = candidate_span_label.split('|')
        candidate_span_label = [(line.split(',')[0], line.split(',')[1].split(' ')[0], line.split(' ')[-1]) for line in
                                candidate_span_label]
        input_id = [word2idx.get(te, word2idx['</UNK>']) for te in text]
        for start, end, label_content in candidate_span_label:
            start_id[int(start)] = label2index[label_content]
            end_id[int(end)] = label2index[label_content]

        if len(input_id) > max_seq_lenth:
            input_id = input_id[:max_seq_lenth]
            start_id = start_id[:max_seq_lenth]
            end_id = end_id[:max_seq_lenth]
            input_mask = input_mask[:max_seq_lenth]

        while len(input_id) < max_seq_lenth:
            input_id.append(0)
            start_id.append(0)
            end_id.append(0)
            input_mask.append(0)

        # print(len(input_id),len(start_id),len(end_id),len(input_mask))
        assert len(input_id) == len(start_id) == len(end_id) == len(input_mask) == max_seq_lenth
        INPUT_ID.append(input_id)
        start_pos.append(start_id)
        end_pos.append(end_id)
        INPUT_MASK.append(input_mask)

    return INPUT_ID, INPUT_MASK, start_pos, end_pos

