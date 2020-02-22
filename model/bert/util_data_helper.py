import logging
import os
from util.util import compute_spans_bieos

logger = logging.getLogger(__name__)


cys_label = {"0": "O", "1": "RT", "2": "LOC", "3": 'PER', "4": "ORG", "5": "SW", "6": "VUL_ID"}


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class DoubleInputFeatures(object):
    """双指针."""

    def __init__(self, input_ids, input_mask, segment_ids,start_ids,end_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.end_ids = end_ids


def Doubue_convert_examples_to_features(
        examples,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    label_map = {j: int(i) for i, j in label_map.items()}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_tok = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.append(word_tokens[0])
                label_tok.append(label)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens

        assert len(tokens) == len(label_tok)
        start_id = [0] * len(tokens)
        end_id = [0] * len(tokens)
        candidate_span_label = compute_spans_bieos(label_tok)
        if len(candidate_span_label) > 0:
            candidate_span_label = candidate_span_label.split('|')
        candidate_span_label = [(line.split(',')[0], line.split(',')[1].split(' ')[0], line.split(' ')[-1]) for line in
                                candidate_span_label]

        for start, end, label_content in candidate_span_label:
            start_id[int(start)] = label_map[label_content]
            end_id[int(end)] = label_map[label_content]


        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_id = start_id[: (max_seq_length - special_tokens_count)]
            end_id = end_id[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        start_id += [pad_token_label_id]
        end_id += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            start_id += [pad_token_label_id]
            end_id += [pad_token_label_id]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            start_id += [pad_token_label_id]
            end_id += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_id = [pad_token_label_id] + start_id
            end_id = [pad_token_label_id] + end_id
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_id += [pad_token_label_id] * padding_length
            end_id += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_id) == max_seq_length
        assert len(end_id) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("start_ids: %s", " ".join([str(x) for x in start_id]))
            logger.info("end_ids: %s", " ".join([str(x) for x in end_id]))

        features.append(
            DoubleInputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, start_ids=start_id,end_ids=end_id)
        )
    return features

    

class MRCInputFeatures(object):
    """MRC query"""

    def __init__(self,unique_id,tokens,input_ids,input_mask,segment_ids,ner_cate,start_position=None,end_position=None,is_impossible=None):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.ner_cate = ner_cate
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def MRC_convert_examples_to_features(
        examples,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    pass

