import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_bert import BertPreTrainedModel,BertModel,BertConfig

class BertForDoublePointClassification(BertPreTrainedModel):
    def __init__(self,args,config):
        super(BertForDoublePointClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.end_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None,attention_mask=None, token_type_ids=None,position_ids=None,head_mask=None,inputs_embeds=None,
                 start_ids=None,end_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)

        outputs = (start_logits,end_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if start_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1

                start_logits = start_logits.view(-1, self.num_labels)[active_loss]
                start_labels = start_ids.view(-1)[active_loss]
                start_loss = loss_fct(start_logits, start_labels)

                end_logits = end_logits.view(-1, self.num_labels)[active_loss]
                end_labels = end_ids.view(-1)[active_loss]
                end_loss = loss_fct(end_logits, end_labels)
            # else:
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (start_loss+end_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
