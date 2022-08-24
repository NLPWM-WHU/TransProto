import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


class RACL_Layer(nn.Module):
    def __init__(self, input_dim, opt):
        super(RACL_Layer, self).__init__()
        self.opt = opt

        self.share_conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1, padding=0),
            nn.ReLU()
        )

        self.private_conv = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.aspect_convs = nn.ModuleList([self.private_conv for i in range(self.opt.hop_num)])
        self.context_convs = nn.ModuleList([self.private_conv for i in range(self.opt.hop_num)])

        self.dropout = torch.nn.Dropout(self.opt.keep_prob)
        self.drop_block = DropBlock2D(block_size=3, drop_prob=self.opt.keep_prob)

    def forward(self, inputs, mask, position):
        batch_size = inputs.shape[0]
        inputs = self.dropout(inputs)

        # Shared Feature
        inputs = self.share_conv(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = self.dropout(inputs)

        # Private Feature
        aspect_input, context_input = list(), list()
        aspect_prob_list, senti_prob_list = list(), list()
        aspect_input.append(inputs)
        context_input.append(inputs)

        # We found that the SC task is more difficult than the AE and OE tasks.
        # Hence, we augment it with a memory-like mechanism by updating the aspect query with the retrieved contexts.
        # Refer to https://www.aclweb.org/anthology/D16-1021/ for for more details about the memory network.
        query = list()
        query.append(inputs)

        for hop in range(self.opt.hop_num):
            # AE & OE Convolution
            aspect_conv = self.aspect_convs[hop](aspect_input[-1].transpose(1, 2)).transpose(1, 2)

            # SC Convolution
            context_conv = self.context_convs[hop](context_input[-1].transpose(1, 2)).transpose(1, 2)

            # SC Aspect-Context Attention
            word_see_context = torch.matmul(query[-1], F.normalize(context_conv, p=2, dim=-1).transpose(1, 2)) * position
            word_att_context = self.softmask_2d(word_see_context, mask, scale=True)

            # Relation R2 & R3
            query_now = (query[-1] + torch.matmul(word_att_context, context_conv)) # query + value
            query.append(query_now) # update query

            # SC Prediction

            # Stacking

            aspect_input.append(self.dropout(aspect_conv))
            context_input.append(self.dropout(context_conv))
            # aspect_input.append(self.drop_block(aspect_conv.unsqueeze(-1)).squeeze(-1))
            # context_input.append(self.drop_block(context_conv.unsqueeze(-1)).squeeze(-1))

        return aspect_input[-1], query[-1]

    def softmask_2d(self, x, mask, scale=False):
        if scale == True:
            max_value, _ = torch.max(x, dim=-1, keepdim=True)
            x -= max_value
        length = mask.shape[1]

        mask_d1 = (torch.unsqueeze(mask, 1)).repeat([1, length, 1])

        y = torch.exp(x) *  mask_d1

        sumx = torch.sum(y, dim=-1, keepdim=True)
        att = y / (sumx + 1e-10)

        mask_d2 =(torch.unsqueeze(mask, 2)).repeat([1, 1, length])
        att *= mask_d2
        return att