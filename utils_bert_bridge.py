from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer

class ABSATokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels_a, labels_s, proto_words, proto_probs): # for AE

        '''
        Tokens : Original words from review sentences
        Words : Supplementary words from prototypes
        '''

        'token & label'
        sub_tokens_all, split_labels_a, split_labels_s, idx_map_tokens = [], [], [], []
        for ix, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            sub_tokens_all.extend(sub_tokens)
            for jx, sub_token in enumerate(sub_tokens):
                if labels_a[ix] == [0, 1, 0] and jx > 0: # if subword is 'B', it should be tokenized as 'B I I ...'
                    split_labels_a.append([0, 0, 1])
                else:
                    split_labels_a.append(labels_a[ix])
                split_labels_s.append(labels_s[ix])
                idx_map_tokens.append(ix)

        # tokened_review = ' '.join(sub_tokens_all)
        word_proto_mat = np.zeros([len(idx_map_tokens)+1, len(idx_map_tokens)+1])
        for row_pre, col_pre in enumerate(idx_map_tokens):
            row = row_pre + 1 # for [CLS] token
            col = col_pre
            word_proto_mat[row][col] = 1.



        'prototype'
        sub_proto_words_all = []
        # sub_proto_maxnum = 10
        for proto_topk in (proto_words):
            sub_proto_words = []
            for proto_k in proto_topk:
                sub_proto = self.wordpiece_tokenizer.tokenize(proto_k)
                sub_proto_words.append(sub_proto)
            sub_proto_words_all.append(sub_proto_words)

        # sub_words_all, split_words, idx_map_protos, sub_probs_all = [], [], [], []
        # for ix, word in enumerate(proto_words):
        #     sub_words = self.wordpiece_tokenizer.tokenize(word)
        #     sub_words_all.extend(sub_words)
        #     for sub_word in sub_words:
        #         sub_probs_all.append(proto_probs[ix])
        #         idx_map_protos.append(ix)
        #
        # 'create a word-proto matrix'
        # word_proto_mat= np.zeros([len(idx_map_tokens)+1, len(idx_map_protos)+1]) # add a element for [CLS]
        #
        # for row_pre, idx_token in enumerate(idx_map_tokens):
        #     row = row_pre + 1
        #     for col_pre, idx_proto in enumerate(idx_map_protos):
        #         col = col_pre + 1
        #         if idx_token == idx_proto:
        #             word_proto_mat[row][col] = 1.0
        #
        # word_proto_mat = softmax(word_proto_mat)

        return sub_tokens_all, split_labels_a, split_labels_s, sub_proto_words_all, proto_probs, word_proto_mat #, idx_map, tokened_review

class ABSADataset():
    def __init__(self, process, fname, opt, unlabel_fname=None):
        data = []

        print('processing {} files: {}'.format(process, fname))
        self.read_data(fname, opt, data, data_type=process)

        if unlabel_fname is not None:
            print('processing unlabeled files: {}'.format(unlabel_fname))
            self.read_data(unlabel_fname, opt, data, data_type='unlabel')

        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def read_data(self, fname, opt, data, data_type):
        max_length = opt.max_sentence_len
        source = opt.source
        target = opt.target
        topk = opt.topk
        tau = opt.tau

        assert data_type in ['train', 'test', 'unlabel', 'dev']

        # with open('./data/dep.dict', 'r', encoding='utf-8') as f:
        #     dep_dict = eval(f.read())
        # with open('./data/pos.dict', 'r', encoding='utf-8') as f:
        #     pos_dict = eval(f.read())

        'SOURCE TRAINING DATA'
        review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
        # pos_data = open(fname + r'pos.txt', 'r', encoding='utf-8').readlines()
        # graph_data = pickle.load(open(fname + r'dep.graph', 'rb'))
        if data_type in ['train', 'test', 'dev']:
            ae_data = open(fname + r'aspect.txt', 'r', encoding='utf-8').readlines()
            # oe_data = open(fname + r'opinion.txt', 'r', encoding='utf-8').readlines()
            sa_data = open(fname + r'polarity.txt', 'r', encoding='utf-8').readlines()
        if data_type in ['train', 'dev']:
            lm_data = open(fname + r'{}-to-{}-prototype-tau{}.txt'.format(source, target, tau), 'r', encoding='utf-8').readlines()
        elif data_type in ['unlabel', 'test']:
            lm_data = open(fname + r'{}-to-{}-prototype-tau{}.txt'.format(target, target, tau), 'r', encoding='utf-8').readlines()

        for index, _ in enumerate(review):

            '''WORD INDEX & MASK'''
            data_per = review[index].lower().strip().split()

            '''ASPECT & OPINION LABELS'''
            if data_type in ['train', 'test', 'dev']:
                ae_labels = ae_data[index].strip().split()
                aspect_label = []
                for l in ae_labels:
                    l = int(l)
                    if l == 0:
                        aspect_label.append([1, 0, 0])
                    elif l == 1:
                        aspect_label.append([0, 1, 0])
                    elif l == 2:
                        aspect_label.append([0, 0, 1])
                    else:
                        raise ValueError

                aspect_y_per = aspect_label

                # oe_labels = oe_data[index].strip().split()
                # opinion_label = []
                # for l in oe_labels:
                #     l = int(l)
                #     if l == 0:
                #         opinion_label.append([1, 0, 0])
                #     elif l == 1:
                #         opinion_label.append([0, 1, 0])
                #     elif l == 2:
                #         opinion_label.append([0, 0, 1])
                #     else:
                #         raise ValueError
                #
                # opinion_y_per = opinion_label

                sa_labels = sa_data[index].strip().split()
                sentiment_label = []
                for l in sa_labels:
                    l = int(l)
                    if l == 0:
                        sentiment_label.append([0, 0, 0])# In testing, we don't know the location of aspect terms.
                    elif l == 1:
                        sentiment_label.append([1, 0, 0])
                    elif l == 2:
                        sentiment_label.append([0, 1, 0])
                    elif l == 3:
                        sentiment_label.append([0, 0, 1])
                    elif l == 4:
                        sentiment_label.append([0, 0, 0])
                    else:
                        raise ValueError
                sentiment_y_per = sentiment_label
            elif data_type in ['unlabel']:
                aspect_y_per = [[0, 0, 0]] * (len(data_per))
                # opinion_y_per = [[0, 0, 0]] * (len(data_per))
                sentiment_y_per = [[0, 0, 0]] * (len(data_per))

            'DOMAIN LABELS'
            if data_type in ['train', 'dev']:
                domain_y_per = [0, 1]
            elif data_type in ['unlabel', 'test']:
                domain_y_per = [1, 0]

            'PROTOTYPES'
            lmwords_list = []
            lmprobs_list = []

            if lm_data[index].strip() == 'NULL':
                raise ValueError
            else:
                segments = lm_data[index].strip().split('###')
                for segment in segments:
                    lminfo = segment.split('@@@')
                    try:
                        position = int(lminfo[0])
                    except ValueError:
                        print(lm_data[index].strip())
                        print('debug')
                    pairs = lminfo[1:]
                    words = []
                    probs = []
                    topk_cnt = 0
                    for pair in pairs:
                        if topk_cnt >= topk:
                            break
                        word = pair.split()[0]
                        prob = float(pair.split()[1])
                        words.append(word)
                        probs.append(prob)
                        topk_cnt += 1

                    lmwords_list.append(words)
                    lmprobs_list.append(softmax(probs))

            lmwords_per = lmwords_list
            lmprobs_per = lmprobs_list

            data_per = {'x': data_per,
                        'proto_words': lmwords_per,
                        'proto_probs': lmprobs_per,
                        'aspect_y': aspect_y_per,
                        # 'opinion_y': opinion_y_per,
                        'sentiment_y': sentiment_y_per,
                        'domain_y': domain_y_per}

            bert_per = self.convert_data_to_bert_type(data_per, opt, data_type)

            # if 'device' in fname and data_type == 'train':  # only use valid training samples in DEVICE
            #     print('Source Domain is DEVICE. Only use samples containing aspects for training.')
            #     if [0, 1, 0] in aspect_y_per:
            #         data.append(bert_per)
            #     else:
            #         continue
            # else:
            #     data.append(bert_per)
            data.append(bert_per)

    def convert_data_to_bert_type(self, data_per, opt, data_type):
        tokenizer = opt.tokenizer
        max_length = opt.max_sentence_len

        tokens, labels_aspect, labels_sentiment, proto_words, proto_probs, word_proto_mat = tokenizer.subword_tokenize(data_per['x'],
                                                                                                     data_per['aspect_y'],
                                                                                                     data_per['sentiment_y'],
                                                                                                     data_per['proto_words'],
                                                                                                     data_per['proto_probs'])

        if len(tokens) > max_length - 2:
            print('The sentence exceeds the maximum length!')
            raise ValueError

        'Generate Bert Inputs'
        bert_tokens = []
        bert_tokens.append("[CLS]")

        bert_segment_ids = []
        bert_segment_ids.append(0)

        bert_label_onehot_a = []
        bert_label_onehot_a.append([0, 0, 0])

        bert_label_onehot_s = []
        bert_label_onehot_s.append([0, 0, 0])

        for token_idx, token in enumerate(tokens):
            bert_tokens.append(token)
            bert_segment_ids.append(0)
            bert_label_onehot_a.append(labels_aspect[token_idx])
            bert_label_onehot_s.append(labels_sentiment[token_idx])

        bert_tokens.append("[SEP]")
        bert_segment_ids.append(0)
        bert_label_onehot_a.append([0, 0, 0])
        bert_label_onehot_s.append([0, 0, 0])

        bert_input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        bert_input_mask = [1] * len(bert_input_ids)

        valid_length = len(bert_input_ids)

        # Zero-pad up to the sequence length.
        while len(bert_input_ids) < max_length:
            bert_input_ids.append(0)
            bert_input_mask.append(0)
            bert_segment_ids.append(0)
            bert_label_onehot_a.append([0, 0, 0])
            bert_label_onehot_s.append([0, 0, 0])

        bert_label_id_a = []
        for label_onehot_a_per in bert_label_onehot_a:
            if label_onehot_a_per == [0, 0, 0]:
                bert_label_id_a.append(-1)
            else:
                bert_label_id_a.append(label_onehot_a_per.index(1))

        bert_label_id_s = []
        for label_onehot_s_per in bert_label_onehot_s:
            if label_onehot_s_per == [0, 0, 0]:
                bert_label_id_s.append(-1)
            else:
                bert_label_id_s.append(label_onehot_s_per.index(1))

        'Prototypes'
        bert_protos = []
        sub_proto_maxnum = 15
        for proto_topk in proto_words:
            proto_topk_ids = []
            for proto_k in proto_topk:
                proto_k_ids = tokenizer.convert_tokens_to_ids(proto_k)
                proto_topk_ids.append(proto_k_ids + [0] * (sub_proto_maxnum - len(proto_k_ids)))
            bert_protos.append(proto_topk_ids)
        bert_protos = np.array(bert_protos)
        # Zero-pad up to the sequence length.
        # while len(bert_proto_ids) < max_length:
        #     bert_proto_ids.append(0)
        try:
            bert_protos_pad = np.pad(bert_protos, ((0, max_length-bert_protos.shape[0]), (0, 0), (0, 0)), 'constant')
        except ValueError:
            print('debug')
        word_proto_mat_pad = np.pad(word_proto_mat,
                                    ((0, max_length-word_proto_mat.shape[0]), (0, max_length-word_proto_mat.shape[1])),
                                    'constant')
        bert_proto_probs = np.array(proto_probs)
        bert_proto_probs_pad = np.pad(bert_proto_probs, ((0, max_length-bert_proto_probs.shape[0]), (0, 0)), 'constant')

        position_mat = position_matrix(valid_length, max_length)

        bert_label_id_d = data_per['domain_y'].index(1)

        assert len(bert_input_ids) == max_length
        assert len(bert_input_mask) == max_length
        assert len(bert_segment_ids) == max_length
        assert len(bert_label_onehot_a) == max_length
        assert len(bert_label_id_a) == max_length
        assert len(bert_label_onehot_s) == max_length
        assert len(bert_label_id_s) == max_length
        assert bert_protos_pad.shape[0] == max_length
        assert word_proto_mat_pad.shape == (max_length, max_length)
        assert position_mat.shape == (max_length, max_length)

        bert_per = {'input_ids': np.array(bert_input_ids, dtype='int64'),
                    'segment_ids': np.array(bert_segment_ids, dtype='int64'),
                    'input_mask': np.array(bert_input_mask, dtype='int64'),
                    'proto_ids': np.array(bert_protos_pad, dtype='int64'),
                    'proto_probs': np.array(bert_proto_probs_pad, dtype='float32'),
                    'word_prob_mat': np.array(word_proto_mat_pad, dtype='float32'),
                    'position_mat': np.array(position_mat, dtype='float32'),
                    'label_id_a': np.array(bert_label_id_a, dtype='int64'),
                    'label_onehot_a': np.array(bert_label_onehot_a, dtype='int64'),
                    'label_id_s': np.array(bert_label_id_s, dtype='int64'),
                    'label_onehot_s': np.array(bert_label_onehot_s, dtype='int64'),
                    'label_id_d': bert_label_id_d}

        return bert_per

'GRL IMPLEMENTATION'

from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    batch_size, seq_length, prob_dim = logits.shape

    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y


    y = y.view(-1, prob_dim)

    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind.view(-1, 1), 1)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(batch_size, seq_length, prob_dim)

# def softmax(probs):
#     probs = np.array(probs)
#     mask = np.asarray(probs != 0, np.float32)
#     probs -= np.max(probs, axis=-1, keepdims=True)
#     probs_exp = np.exp(probs) * mask
#     return probs_exp / (np.sum(probs_exp, axis=-1, keepdims=True) + 1e-6)

# def softmax(probs):
#     probs = np.array(probs)
#     probs -= np.max(probs, axis=-1, keepdims=True)
#     return np.exp(probs) / (np.sum(np.exp(probs), axis=-1, keepdims=True) + 1e-6)

def softmax(probs):
    probs = np.array(probs)
    # probs = probs ** (1 / 0.3)
    mask = np.asarray(probs != 0, np.float32)
    probs -= np.max(probs, axis=-1, keepdims=True)
    probs_exp = np.exp(probs) * mask
    att = probs_exp / (np.sum(probs_exp, axis=-1, keepdims=True) + 1e-6)
    return att
# def refine_softmax(probs):
#     refined_scores = probs ** (1 / 0.3)
#     refined_probs = refined_scores/(np.sum(refined_scores, -1, keepdims=True) + 1e-6)
#     return refined_probs

def position_matrix(sen_len, max_len):
    a = np.zeros([max_len, max_len], dtype=np.float32)

    for i in range(sen_len):
        for j in range(sen_len):
            if i == j:
                a[i][j] = 0.
            else:
                a[i][j] = 1/(np.log2(2 + abs(i - j)))
                # a[i][j] = 1/(abs(i - j))

    return a

'GRL IMPLEMENTATION'

from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None