from torch import nn
from transformers import BertModel
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import subprocess as sp
import config
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from AsymetricLoss import AsymmetricLoss


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(sp.check_output(command.split()).decode('ascii'))
    return memory_free_values


def convert_weights(roberta_base, roberta_sentiment, params):
    target = dict(roberta_base.named_parameters())
    source = dict(roberta_sentiment.named_parameters())
    for part in params:
        target[part].data.copy_(source[part].data)


class BertClassifier(nn.Module):

    def __init__(self, output_size, dropout=0.5, hidden_size=768, train_language=True, two_bert='ba_atten',
                 model=None):
        super(BertClassifier, self).__init__()
        if not model:
            # self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=False,
            #                                       output_attentions=False, num_labels=31)
            self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=False,
                                                            output_attentions=False, num_labels=config.num_labels)
        else:
            self.model = model
            self.bert = AutoModelForSequenceClassification.from_pretrained("roberta-base",
                                                                           num_labels=config.num_labels)  # TODO: revise for two inputs
            self.bert.roberta = self.model.roberta
            params = self.bert.state_dict()
            del params['roberta.embeddings.position_ids']
            del params['classifier.out_proj.weight']
            del params['classifier.out_proj.bias']
            convert_weights(self.bert, self.model, params)
            assert torch.equal(self.bert.state_dict()['roberta.embeddings.word_embeddings.weight'],
                               self.model.state_dict()['roberta.embeddings.word_embeddings.weight']) == True
            del self.model
        if train_language:
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.two_bert = two_bert
        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.score_proj = nn.Linear(hidden_dim, 1)
        self.output_size = output_size
        # self.weights = nn.Parameter(torch.rand(13, 1))
        self.fc1 = nn.Linear(hidden_size, self.output_size)
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.weight = nn.Parameter(torch.rand(hidden_size))
        self.V = nn.Parameter(torch.rand(768))
        self.softmax = nn.LogSoftmax(dim=1)
        self.multi_head = nn.MultiheadAttention(embed_dim=768, num_heads=1, dropout=0.5, batch_first=True)
        self.AsymetricLoss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    def dot_attention(self, q, k, v):
        attn_weights = torch.matmul(q, k.transpose(2, 1))  # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, 1)
        # output = torch.matmul(attn_weights, v)  # [bs, poly_m, dim]
        output = attn_weights * v
        return output

    def bahanau_attention(self, query, key, value):
        key_new = key.unsqueeze(1).repeat(1, query.shape[1], 1)
        _cat = torch.cat((query, key_new), dim=2)
        out = torch.tanh(self.proj(_cat))
        out = out.permute(0, 2, 1)
        V = self.V.repeat(query.size(0), 1).unsqueeze(1)
        e = torch.bmm(V, out).squeeze(1)
        att_weights = F.softmax(e, dim=1)
        context = torch.bmm(att_weights.unsqueeze(1), value)
        return context.squeeze(1)

    def concat_layer(self, output_1, output_2):
        pooled_outputs = self.dropout(output_1)
        pooled_outputs2 = self.dropout(output_2)
        output = torch.cat((pooled_outputs, pooled_outputs2), dim=1)
        out = self.proj(output)
        return out

    def self_attention(self, query, key, value):

        outputs = self.multi_head(query, key, value, need_weights=False)[0]
        return outputs[:, 0, :]

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, return_dict=False,
                input_ids_2=None, attention_mask_2=None):

        if config.two_input_flag:
            if self.two_bert == 'ba_atten':
                outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)[0]
                torch.cuda.empty_cache()
                outputs_ctx = self.bert(input_ids_2, attention_mask=attention_mask_2, return_dict=False)[0][:, 0, :]

                context = self.bahanau_attention(outputs, outputs_ctx, outputs)
                logits = self.fc1(context)
            elif self.two_bert == "concat":
                # outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0, :]
                # outputs2 = self.bert(input_ids_2, attention_mask=attention_mask_2, return_dict=False)[0][:, 0, :]
                outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)[1]
                outputs2 = self.bert(input_ids_2, attention_mask=attention_mask_2, return_dict=False)[1]
                context = self.concat_layer(outputs, outputs2)
            elif self.two_bert == 'multi_head':
                outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)[0]
                outputs2 = self.bert(input_ids_2, attention_mask=attention_mask_2, return_dict=False)[0]
                context = self.self_attention(outputs, outputs2, outputs)
                logits = self.fc1(context)


        else:
            logits = self.bert(input_ids, attention_mask,token_type_ids=token_type_ids, return_dict=False)[0]

        loss = None

        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # loss_fct = self.AsymetricLoss()
            loss = self.AsymetricLoss(logits.view(-1, self.output_size), labels.float())

        if not return_dict:
            output = (logits,)  # + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {"loss": loss, "logits": logits, "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions}
