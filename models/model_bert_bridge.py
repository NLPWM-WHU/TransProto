# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.decnn_conv import DECNN_CONV
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel
# from pytorch_pretrained_bert.optimization import BertAdam
from modeling_customize import PreTrainedBertModel, BertModel
# from layers.racl_layer import RACL_Layer_with_BERT
from layers.racl_layer import RACL_Layer
from utils_bert_bridge import ReverseLayerF

class BertForSequenceLabeling(PreTrainedBertModel): # 通过pretrained来实例化一个BERT
    def __init__(self, config, custom_opt, num_labels=3):
        super(BertForSequenceLabeling, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, custom_opt)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        # self.racl = RACL_Layer_with_BERT(config.hidden_size, custom_opt)
        self.racl = RACL_Layer(config.hidden_size, custom_opt)
        self.aspect_classifier = torch.nn.Linear(768*2, num_labels)
        self.sentiment_classifier = torch.nn.Linear(768*2, num_labels)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(768*2, 768*2),
        #     nn.ReLU(),
        #     nn.Linear(768*2, 768*2),
        #     nn.ReLU(),
        #     nn.Linear(768*2, 768*2),
        #     nn.ReLU(),
        #     nn.Linear(768*2, 2),
        # )
        self.domain_classifier_s = nn.Sequential(
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 2),
        )
        self.domain_classifier_a = nn.Sequential(
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 2),
        )

        self.custom_opt = custom_opt

        if custom_opt.use_prototype == 1:
            print('Prototype Used!')
            self.gate = nn.Sequential(
                nn.Linear(768*2, 768*2),
                nn.Sigmoid(),
                         )
        else:
            print('Prototype Not Used!')

    def forward(self, inputs, opt, is_training=False):
        input_ids, segment_ids, input_mask, proto_ids, proto_probs, word_proto_mat, position_mat = inputs
        # sequence_output, _ = self.bert(input_ids, segment_ids, input_mask, proto_ids, proto_probs, word_proto_mat, output_all_encoded_layers=False)
        # sequence_output = self.dropout(sequence_output)

        sequence_input, sequence_proto = self.bert(input_ids, segment_ids, input_mask, proto_ids, proto_probs,
                                                   word_proto_mat, output_all_encoded_layers=False)
        enhanced_emb = self.conditioner(sequence_input, sequence_proto, self.gate)
        enhanced_emb = self.dropout(enhanced_emb)

        enhanced_conv_a, enhanced_conv_s = self.racl(enhanced_emb, input_mask, position_mat)
        # enhanced_conv = torch.cat([enhanced_conv_a, enhanced_conv_s], -1)
        # enhanced_conv = enhanced_conv_s
        # enhanced_conv = enhanced_conv_a

        logits_a = self.aspect_classifier(enhanced_conv_a)# b, 70, 3
        logits_s = self.sentiment_classifier(enhanced_conv_s)

        if is_training:
            'domain'
            # summary, _ = torch.max(enhanced_conv.transpose(2,1), -1)
            # reverse_summary = ReverseLayerF.apply(summary, self.custom_opt.alpha)
            #
            # logits_d = self.domain_classifier(reverse_summary)

            summary_a, _ = torch.max(enhanced_conv_a.transpose(2,1), -1)
            reverse_summary_a = ReverseLayerF.apply(summary_a, self.custom_opt.alpha)

            summary_s, _ = torch.max(enhanced_conv_s.transpose(2,1), -1)
            reverse_summary_s = ReverseLayerF.apply(summary_s, self.custom_opt.alpha)

            logits_da = self.domain_classifier_a(reverse_summary_a)
            logits_ds = self.domain_classifier_s(reverse_summary_s)
            # logits_d = (self.domain_classifier_a(reverse_summary_a) + self.domain_classifier_s(reverse_summary_s)) / 2.



            # return logits_a, logits_s, logits_d
            return logits_a, logits_s, logits_da, logits_ds
        else:
            return logits_a, logits_s

    def conditioner(self, main_emb, auxi_emb, gate):
        concat_emb = torch.cat([main_emb, auxi_emb], -1)
        concat_gate = gate(concat_emb)
        enhanced_emb = concat_emb * concat_gate

        return enhanced_emb



