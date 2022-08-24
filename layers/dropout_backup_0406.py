import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


class RACL_Layer(nn.Module):
    def __init__(self, input_dim, opt):
        super(RACL_Layer, self).__init__()
        self.opt = opt

        # self.share_conv = nn.Sequential(
        #     nn.Conv1d(input_dim, 256, 1, padding=0),
        #     nn.ReLU()
        # )

        self.aspect_convs = nn.ModuleList([nn.Sequential(nn.Conv1d(768, 768, 3, padding=1), nn.ReLU())
                                           for i in range(self.opt.hop_num)])

        self.dropout = torch.nn.Dropout(self.opt.keep_prob)

        self.attention_pooling = nn.Sequential(
                                       nn.Linear(768, 768),
                                       nn.Tanh(),
                                       nn.Linear(768, 1),
                                       )

    def forward(self, inputs, mask, position):
        batch_size = inputs.shape[0]
        # inputs = self.dropout(inputs)

        # Shared Feature
        # inputs = self.share_conv(inputs.transpose(1, 2)).transpose(1, 2)
        # inputs = self.dropout(inputs)

        # Private Feature
        aspect_input = list()
        aspect_input.append(inputs)

        attention_out = []

        word_see_context_pre = torch.matmul(inputs, inputs.transpose(1, 2))
        word_att_context_pre = self.softmask_2d(word_see_context_pre, mask, scale=True)

        # Relation R2 & R3
        attention_out.append(torch.matmul(word_att_context_pre, inputs).unsqueeze(-2))

        for hop in range(self.opt.hop_num):
            # AE & OE Convolution
            aspect_conv = self.aspect_convs[hop](aspect_input[-1].transpose(1, 2)).transpose(1, 2)
            aspect_conv = self.dropout(aspect_conv)

            # SC Convolution
            # context_conv = self.context_convs[hop](context_input[-1].transpose(1, 2)).transpose(1, 2)
            # context_conv = aspect_conv

            # SC Aspect-Context Attention
            word_see_context = torch.matmul(F.normalize(aspect_conv, p=2, dim=-1), F.normalize(aspect_conv, p=2, dim=-1).transpose(1, 2))
            word_att_context = self.softmask_2d(word_see_context, mask, scale=True)

            # Relation R2 & R3
            attention_out.append(torch.matmul(word_att_context, aspect_conv).unsqueeze(-2))
            # attention_out.append(torch.matmul(word_att_context, inputs).unsqueeze(-2))

            aspect_input.append((aspect_conv))

        # attention_result = torch.mean(torch.cat(attention_out, -2), -2)
        attention_result = torch.cat(attention_out, -2) # b, 70, 3, 256
        attention_weight =  F.softmax(self.attention_pooling(attention_result).squeeze(-1), dim=-1) # b, 70, 3
        weighted_result = torch.matmul(attention_weight.unsqueeze(-2), attention_result).squeeze(-2)


        return aspect_input[-1], weighted_result

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
