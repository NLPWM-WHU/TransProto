# -*- coding: utf-8 -*-
from layers.decnn_conv import DECNN_CONV
from layers.GraphConvolution import GraphConvolution
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class DECNN(nn.Module):
    def __init__(self, global_emb, opt):
        super(DECNN, self).__init__()
        self.opt = opt
        global_emb = torch.tensor(global_emb, dtype=torch.float32).to(self.opt.device)
        self.global_emb = nn.Embedding.from_pretrained(global_emb)
        self.global_emb.weight.requires_grad = False
        if opt.ablation == 'pos':
            self.dep_emb = torch.nn.Linear(40, opt.emb_dim, bias=False)
        elif opt.ablation == 'dep':
            self.pos_emb = torch.nn.Linear(45, opt.emb_dim, bias=False)
        else:
            self.pos_emb = torch.nn.Linear(45, opt.emb_dim // 2, bias=False)
            self.dep_emb = torch.nn.Linear(40, opt.emb_dim // 2, bias=False)

        if opt.use_syntactic == 1:
            self.gate_syn = nn.Sequential(
                                      nn.Linear(2 * opt.emb_dim, 2 * opt.emb_dim),
                                      nn.Sigmoid(),
                                     )
        if opt.use_semantic == 1:
            self.gate_sem = nn.Sequential(
                                      nn.Linear(2 * opt.emb_dim, 2 * opt.emb_dim),
                                      nn.Sigmoid(),
                                     )

        if opt.use_syntactic == 1 and opt.use_semantic == 1:
            self.gate_global = nn.Sequential(
                                      nn.Linear(4 * opt.emb_dim, 1),
                                      nn.Sigmoid(),
                                     )

        enhanced_emb_dim_dict = {0: opt.emb_dim,
                                 1: 2 * opt.emb_dim,
                                 2: 4 * opt.emb_dim}

        enhanced_emb_dim = enhanced_emb_dim_dict[opt.use_syntactic + opt.use_semantic]

        self.conv_op_aspect = DECNN_CONV(enhanced_emb_dim, self.opt)
        self.conv_op_sentiment = DECNN_CONV(enhanced_emb_dim, self.opt)

        self.aspect_classifier = torch.nn.Linear(256, self.opt.class_num)
        self.sentiment_classifier = torch.nn.Linear(256, self.opt.class_num)
        self.domain_classifier = nn.Sequential(
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 2),
                                   )

        self.dropout = torch.nn.Dropout(self.opt.keep_prob)

    def forward(self, inputs, alpha=0., is_training=False):
        x, mask, dep, pos, adj_multi, lmwords, lmprobs, positions = inputs
        adj = (adj_multi > 0).float()
        x_emb = self.global_emb(x)

        lm_emb_per = self.global_emb(lmwords)  # b, 70, 3, 400
        lm_prob_per = lmprobs.unsqueeze(2)
        lm_emb = torch.matmul(lm_prob_per, lm_emb_per).squeeze(2)

        if self.opt.use_syntactic == 1:
            if self.opt.ablation == 'pos':
                enhanced_emb_syn = self.conditioner(x_emb, self.dep_emb(dep), self.gate_syn)
            elif self.opt.ablation == 'dep':
                enhanced_emb_syn = self.conditioner(x_emb, self.pos_emb(pos), self.gate_syn)
            else:
                enhanced_emb_syn = self.conditioner(x_emb, torch.cat([self.pos_emb(pos), self.dep_emb(dep)], -1), self.gate_syn)

        if self.opt.use_semantic == 1:
            enhanced_emb_sem = self.conditioner(x_emb, lm_emb, self.gate_sem)


        if self.opt.use_syntactic == 0 and self.opt.use_semantic == 0:
            enhanced_emb = x_emb
        elif self.opt.use_syntactic == 1 and self.opt.use_semantic == 0:
            enhanced_emb = enhanced_emb_syn
        elif self.opt.use_syntactic == 0 and self.opt.use_semantic == 1:
            enhanced_emb = enhanced_emb_sem
        elif self.opt.use_syntactic == 1 and self.opt.use_semantic == 1:
            gate_global = self.gate_global(torch.cat([enhanced_emb_syn, enhanced_emb_sem], -1))
            enhanced_emb = torch.cat([gate_global * enhanced_emb_syn, (1 - gate_global) * enhanced_emb_sem], -1)
        else:
            print('Error in Enhancement !')
            raise ValueError


        enhanced_emb_tran = self.dropout(enhanced_emb).transpose(1, 2)
        enhanced_conv_a = self.conv_op_aspect(enhanced_emb_tran)
        enhanced_conv_s = self.conv_op_sentiment(enhanced_emb_tran)
        enhanced_conv = torch.cat([enhanced_conv_a, enhanced_conv_s], -1)

        prob_a = F.softmax(self.aspect_classifier(enhanced_conv_a), -1) # b, 70, 3
        prob_s = F.softmax(self.sentiment_classifier(enhanced_conv_s), -1)

        if is_training:
            'domain'
            summary, _ = torch.max(enhanced_conv.transpose(2,1), -1)
            reverse_summary = ReverseLayerF.apply(summary, alpha)

            prob_d = F.softmax(self.domain_classifier(reverse_summary), -1)

            return prob_a, prob_s, prob_d
        else:
            return prob_a, prob_s


    def softmask(self, score, mask):
        score_exp = torch.mul(torch.exp(score), mask)
        sumx = torch.sum(score_exp, dim=-1, keepdim=True)
        return score_exp / (sumx + 1e-5)

    def conditioner(self, main_emb, auxi_emb, gate):
        concat_emb = torch.cat([main_emb, auxi_emb], -1)
        concat_gate = gate(concat_emb)
        enhanced_emb = concat_emb * concat_gate

        return enhanced_emb

