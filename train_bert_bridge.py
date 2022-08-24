# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import argparse
import math
# import os
import sys
from time import strftime, localtime, time
import random
import numpy
from evaluation import *
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from utils_bert_bridge import ABSADataset
from utils_bert_bridge import ABSATokenizer

from pytorch_pretrained_bert.optimization import BertAdam
from models.model_bert_bridge import BertForSequenceLabeling
# from models.model_mlm_0522 import BertForSequenceLabeling

print('Model is model_bertpt_0520')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if opt.use_unlabel == 1:
            self.train_data = ABSADataset('train', opt.dataset_file['train'], opt, opt.dataset_file['unlabel'])
        else:
            self.train_data = ABSADataset('train', opt.dataset_file['train'], opt)
        self.test_data = ABSADataset('test', opt.dataset_file['test'], opt)
        self.dev_data = ABSADataset('dev', opt.dataset_file['dev'], opt)

        # self.num_train_steps = int(len(self.train_data) / self.opt.batch_size) * self.opt.num_epoch
        self.num_train_steps = int(len(self.train_data) / self.opt.batch_size) * 50


        'Model Alignment'
        self.model = BertForSequenceLabeling.from_pretrained(self.opt.bert_path, custom_opt=opt, num_labels=opt.class_num)
        self.model.cuda()

        # self.MLM_model = BertForMaskedLM.from_pretrained(self.opt.bert_path)
        # self.MLM_model.cuda()


        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        # for child in self.model.children():
        #     if type(child) != BertForSequenceLabeling:  # skip bert params
        #         for p in child.parameters():
        #             if p.requires_grad:
        #                 if len(p.shape) > 1:
        #                     stdv = 1. / math.sqrt(p.shape[0])
        #                     torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        #                 else:
        #                     stdv = 1. / math.sqrt(p.shape[0])
        #                     torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        for name, para in self.model.named_parameters():
            if not name.startswith('bert'):
                print(name)
                if para.requires_grad:
                    stdv = 1. / math.sqrt(para.shape[0])
                    torch.nn.init.uniform_(para, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, test_data_loader, dev_data_loader, is_training):
        max_dev_metric = 0.
        min_dev_loss = 1000.
        global_step = 0
        path = None
        aspect_f1_list, opinion_f1_list, sentiment_acc_list, sentiment_f1_list, ABSA_f1_list = list(), list(), list(), list(), list()
        dev_metric_list, dev_loss_list = list(), list()

        for epoch in range(self.opt.num_epoch):
            'TRAIN'
            epoch_start = time()
            n_correct, n_total, loss_total = 0, 0, 0
            aspect_loss_total, opinion_loss_total, sentiment_loss_total, domain_loss_total, reg_loss_total = 0., 0., 0., 0., 0.
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):

                inputs = [sample_batched[col].to(self.opt.device) for col in ['input_ids',
                                                                              'segment_ids',
                                                                              'input_mask',
                                                                              'proto_ids',
                                                                              'proto_probs',
                                                                              'word_prob_mat',
                                                                              'position_mat']]

                label_id_a, label_onehot_a, label_id_s, label_onehot_s, label_id_d = [sample_batched[col].to(self.opt.device)
                                                                          for col in ['label_id_a','label_onehot_a',
                                                                         'label_id_s','label_onehot_s','label_id_d']]

                logits_a, logits_s, logits_da, logits_ds = self.model(inputs, self.opt, is_training=True)

                labels = label_id_a.cpu().numpy()
                label_masks = (labels != -1) + 0

                length = np.sum(label_masks)
                batch_size = (inputs[1]).shape[0]

                aspect_loss = criterion(logits_a.view(-1, self.opt.class_num), label_id_a.view(-1))
                sentiment_loss = criterion(logits_s.view(-1, self.opt.class_num), label_id_s.view(-1))

                opinion_loss = torch.tensor(0.)
                reg_loss = torch.tensor(0.)

                if self.opt.use_unlabel == 1:
                    domain_loss = criterion(logits_da.view(-1, 2), label_id_d.view(-1)) + criterion(logits_ds.view(-1, 2), label_id_d.view(-1))
                    loss = aspect_loss + sentiment_loss + domain_loss
                else:
                    domain_loss = torch.tensor(0.)
                    loss = aspect_loss + sentiment_loss# + domain_loss#+ self.opt.l2_reg * reg_loss
                loss.backward()

                lr_this_step = self.opt.learning_rate * warmup_linear(global_step / self.t_total, self.opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                n_total += length
                loss_total += loss.item() * length
                aspect_loss_total += aspect_loss.item() * length
                opinion_loss_total += opinion_loss.item() * length
                sentiment_loss_total += sentiment_loss.item() * length
                domain_loss_total += domain_loss.item() * batch_size
                reg_loss_total += reg_loss.item() * length

            train_loss = loss_total / n_total
            train_aspect_loss = aspect_loss_total / n_total
            train_opinion_loss = opinion_loss_total / n_total
            train_sentiment_loss = sentiment_loss_total / n_total
            train_domain_loss = domain_loss_total / (len(train_data_loader.dataset) * 2)
            train_reg_loss = reg_loss_total / n_total

            'DEV'
            dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1, \
            dev_loss, dev_aspect_loss, dev_opinion_loss, dev_sentiment_loss, dev_reg_loss = \
            self._evaluate_acc_f1(criterion, dev_data_loader)
            dev_metric = dev_ABSA_f1
            if epoch < 0:
                dev_metric_list.append(0.)
                dev_loss_list.append(1000.)
            else:
                dev_metric_list.append(dev_metric)
                dev_loss_list.append(dev_loss)

            save_indicator = 0
            if (dev_metric > max_dev_metric or dev_loss < min_dev_loss) and epoch >= 100:
                if dev_metric > max_dev_metric:
                    save_indicator = 1
                    max_dev_metric = dev_metric
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss

            'TEST'
            test_aspect_f1, test_opinion_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1, \
            test_loss, test_aspect_loss, test_opinion_loss, test_sentiment_loss, test_reg_loss = \
            self._evaluate_acc_f1(criterion, test_data_loader, epoch, testing=True)
            aspect_f1_list.append(test_aspect_f1)
            opinion_f1_list.append(test_opinion_f1)
            sentiment_acc_list.append(test_sentiment_acc)
            sentiment_f1_list.append(test_sentiment_f1)
            ABSA_f1_list.append(test_ABSA_f1)

            'EPOCH INFO'
            epoch_end = time()
            epoch_time = 'Epoch Time: {:.0f}m {:.0f}s'.format((epoch_end - epoch_start) // 60, (epoch_end - epoch_start) % 60)
            logger.info('\n{:-^80}'.format('Iter' + str(epoch)))
            logger.info('Train: final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, domain loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                        format(train_loss, train_aspect_loss, train_opinion_loss, train_domain_loss, train_sentiment_loss, train_reg_loss, global_step))
            logger.info('Dev:   final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                        format(dev_loss, dev_aspect_loss, dev_opinion_loss, dev_sentiment_loss, dev_reg_loss, global_step))
            logger.info('Dev:   aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                        .format(dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1))
            logger.info('Test:  aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                        .format(test_aspect_f1, test_opinion_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1))
            logger.info('Current Max Metrics Index : {} Current Min Loss Index : {} {}'
                        .format(dev_metric_list.index(max(dev_metric_list)), dev_loss_list.index(min(dev_loss_list)), epoch_time))

        'SUMMARY'
        logger.info('\n{:-^80}'.format('Mission Complete'))
        max_dev_index = dev_metric_list.index(max(dev_metric_list))
        logger.info('Dev Max Metrics Index: {}'.format(max_dev_index))
        logger.info('aspect f1={:.2f}, opinion f1={:.2f}, sentiment acc={:.2f}, sentiment f1={:.2f}, ABSA f1={:.2f},'
                    .format(aspect_f1_list[max_dev_index]*100, opinion_f1_list[max_dev_index]*100, sentiment_acc_list[max_dev_index]*100,
                            sentiment_f1_list[max_dev_index]*100, ABSA_f1_list[max_dev_index]*100))

        min_dev_index = dev_loss_list.index(min(dev_loss_list))
        logger.info('Dev Min Loss Index: {}'.format(min_dev_index))
        logger.info('aspect f1={:.2f}, opinion f1={:.2f}, sentiment acc={:.2f}, sentiment f1={:.2f}, ABSA f1={:.2f},'
                    .format(aspect_f1_list[min_dev_index]*100, opinion_f1_list[min_dev_index]*100, sentiment_acc_list[min_dev_index]*100,
                            sentiment_f1_list[min_dev_index]*100, ABSA_f1_list[min_dev_index]*100))

        return path

    def _evaluate_acc_f1(self, criterion, data_loader, epoch=0, testing=False):
        n_correct, n_total, loss_total = 0, 0, 0
        aspect_loss_total, opinion_loss_total, sentiment_loss_total, reg_loss_total = 0., 0., 0., 0.
        # t_aspect_y_all, t_outputs_all, t_mask_all = None, None, None
        t_aspect_y_all, t_aspect_outputs_all, t_mask_all = list(), list(), list()
        t_opinion_y_all, t_opinion_outputs_all = list(), list()
        t_sentiment_y_all, t_sentiment_outputs_all = list(), list()
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, sample_batched in enumerate(tqdm(data_loader)):

                inputs = [sample_batched[col].to(self.opt.device) for col in ['input_ids',
                                                                              'segment_ids',
                                                                              'input_mask',
                                                                              'proto_ids',
                                                                              'proto_probs',
                                                                              'word_prob_mat',
                                                                              'position_mat']]

                label_id_a, label_onehot_a, label_id_s, label_onehot_s = [sample_batched[col].to(self.opt.device) for col in ['label_id_a',
                                                                                                                              'label_onehot_a',
                                                                                                                              'label_id_s',
                                                                                                                              'label_onehot_s']]
                # with torch.no_grad():
                #     MLM_ids = torch.argmax(self.MLM_model(input_ids, segment_ids, input_mask), -1)

                logits_a, logits_s = self.model(inputs, self.opt)

                labels = label_id_a.cpu().numpy()
                label_masks = (labels != -1) + 0

                preds_a = F.softmax(logits_a, -1).cpu().numpy()
                label_onehot_a = label_onehot_a.cpu().numpy()

                pred_s = F.softmax(logits_s, -1).cpu().numpy()
                label_onehot_s = label_onehot_s.cpu().numpy()

                length = np.sum(label_masks)

                aspect_loss = criterion(logits_a.view(-1, self.opt.class_num), label_id_a.view(-1))
                sentiment_loss = criterion(logits_s.view(-1, self.opt.class_num), label_id_s.view(-1))
                opinion_loss = torch.tensor(0.)
                reg_loss = torch.tensor(0.)
                loss = aspect_loss + sentiment_loss + self.opt.l2_reg * reg_loss

                # if testing:
                    # print('debug')

                n_total += length
                loss_total += loss.item() * length
                aspect_loss_total += aspect_loss.item() * length
                opinion_loss_total += opinion_loss.item() * length
                sentiment_loss_total += sentiment_loss.item() * length
                reg_loss_total += reg_loss.item() * length

                t_aspect_y_all.extend(label_onehot_a.tolist())
                t_aspect_outputs_all.extend(preds_a.tolist())

                t_sentiment_y_all.extend(label_onehot_s.tolist())
                t_sentiment_outputs_all.extend(pred_s.tolist())

                t_mask_all.extend(label_masks.tolist())
            t_loss = loss_total / n_total
            t_aspect_loss = aspect_loss_total / n_total
            t_opinion_loss = opinion_loss_total / n_total
            t_sentiment_loss = sentiment_loss_total / n_total
            t_reg_loss = reg_loss_total / n_total

        t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1 = get_metric(t_aspect_y_all, t_aspect_outputs_all,
                                           np.zeros_like(t_aspect_y_all),  np.zeros_like(t_aspect_y_all),
                                           t_sentiment_y_all,  t_sentiment_outputs_all,
                                           t_mask_all, 1)
        # t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1 = 0.,0.,0.,0.,0.

        # for case study
        # if epoch == 4:
        #     self.prediction_write_to_file(t_aspect_outputs_all, t_mask_all, 'aspect')
        #     self.prediction_write_to_file(t_aspect_y_all, t_mask_all, 'label')

        return t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1, \
               t_loss.item(), t_aspect_loss.item(), t_opinion_loss.item(), t_sentiment_loss.item(), t_reg_loss.item()

    def prediction_write_to_file(self, outputs, masks, term):

        a_preds = np.array(outputs)
        final_mask = np.array(masks)

        a_preds = np.argmax(a_preds, axis=-1)
        # logger.info(np.shape(a_preds))
        # logger.info(np.shape(final_mask))

        aspect_out = []

        for s_idx, sentence in enumerate(a_preds):
            aspect_iter = []
            for w_idx, word in enumerate(sentence):
                if w_idx == 0:
                    continue
                else:
                    if final_mask[s_idx][w_idx] == 0.:
                        break
                    if word == 0:
                        aspect_iter.append(0)
                    elif word == 1:
                        aspect_iter.append(1)
                    elif word == 2:
                        aspect_iter.append(2)
            aspect_out.append(aspect_iter)

        # logger.info(aspect_out)
        aspect_txt = open('data/{}/test{}/{}_{}2{}_pred_{}.txt'.
                          format(self.opt.target, self.opt.split, self.opt.name, self.opt.source[0].upper(), self.opt.target[0].upper(), term),
                          'w', encoding='utf-8')

        for sentence in aspect_out:
            for idx, word in enumerate(sentence):
                if idx == len(sentence) - 1:
                    aspect_txt.write(str(word) + '\n')
                else:
                    aspect_txt.write(str(word) + ' ')


    def run(self):
        # Loss and Optimizer
        # criterion = nn.CrossEntropyLoss()
        # _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.lr_decay)
        # criterion = nn.CrossEntropyLoss(ignore_index=-1)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad == True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.t_total = self.num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.opt.learning_rate,
                             warmup=self.opt.warmup_proportion,
                             t_total=self.t_total)

        train_data_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.test_data, batch_size=self.opt.batch_size, shuffle=False)
        dev_data_loader = DataLoader(dataset=self.dev_data, batch_size=self.opt.batch_size, shuffle=False)

        # self._reset_params()
        _ = self._train(criterion, optimizer, train_data_loader, test_data_loader, dev_data_loader, is_training=True)



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='device', type=str, help='laptop device restaurant service')
    parser.add_argument('--target', default='restaurant', type=str, help='laptop device restaurant service')
    parser.add_argument('--use_unlabel', default=1, type=int, help='use unlabel samples')
    parser.add_argument('--use_prototype', default=1, type=int, help='use unlabel samples')
    parser.add_argument('--split', default=1, type=int, help='specify a split, 1, 2, 3')
    parser.add_argument('--model_name', default='BERT', type=str)
    parser.add_argument('--batch_size', default=16, type=int, help='number of example per batch')
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='learning rate')
    # parser.add_argument('--bert_lr', default=3e-5, type=float, help='learning rate')
    # parser.add_argument('--racl_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=1e-5, type=float, help='learning rate decay')
    parser.add_argument('--num_epoch', default=15, type=int, help='training iteration')
    parser.add_argument('--emb_dim', default=400, type=int, help='dimension of word embedding')
    parser.add_argument('--hidden_dim', default=400, type=int, help='dimension of position embedding')
    parser.add_argument('--keep_prob', default=0.5, type=float, help='dropout keep prob')
    parser.add_argument('--l2_reg', default=1e-5, type=float, help='l2 regularization')
    parser.add_argument('--interpolation', default=0.1, type=float, help='interpolation')
    parser.add_argument('--alpha', default=0.1, type=float, help='domain adversarial training')
    parser.add_argument('--kernel_size', default=5, type=int, help='kernel size')
    parser.add_argument('--hop_num', default=4, type=int, help='hop number')
    parser.add_argument('--class_num', default=3, type=int, help='class number')
    parser.add_argument('--cluster_num', default=200, type=int, help='class number')
    parser.add_argument('--cate_num', default=20, type=int, help='category number') # 20-RES16
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--use_retrieve', default=1, type=int, help='use retrieved samples')
    parser.add_argument('--dynamic_gate', default=1, type=int, help='use dynamic gate for harmonic prediction')
    parser.add_argument('--valset_num', default=0, type=int, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--reuse_embedding', default=1, type=int, help='reuse word embedding & id, True or False')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='uniform_', type=str)
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--ablation', default='MLM', type=str, help='forward_lm, backward_lm, concat, gate, None')
    parser.add_argument('--topk', default=10, type=int, help='1~10')
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--bert_type', default='cross', type=str)
    parser.add_argument('--name', default='BERT-unlabel', type=str)
    # parser.add_argument('--tau', default=10, type=int, help='0~10')
    opt = parser.parse_args()
    start_time = time()
    print('Remote Check Success')
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model_classes = {
        'BERT': BertForSequenceLabeling
    }
    opt.dataset_file = {
            'train': './data/{}/train/'.format(opt.source),
            'dev': './data/{}/test/'.format(opt.source),
            'unlabel': './data/{}/train/'.format(opt.target),
            'test': './data/{}/test/'.format(opt.target)
    }
    input_colses = {
        'DECNN': ['sentence', 'mask', 'position', 'keep_prob'],
        'BERT': ['sentence', 'mask', 'position', 'keep_prob']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
        'kaiming_uniform_':  torch.nn.init.kaiming_uniform_,
        'uniform_':  torch.nn.init.uniform_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    opt.max_sentence_len = 120


    opt.bert_path = './bert/{}'.format(opt.bert_type)
    opt.tokenizer = ABSATokenizer.from_pretrained(opt.bert_path)


    pair = '{}-to-{}'.format(opt.source, opt.target)
    pair_index = {
    'service-to-restaurant': 0,
    'laptop-to-restaurant': 1,
    'device-to-restaurant': 2,
    'restaurant-to-service': 3,
    'laptop-to-service': 4,
    'device-to-service': 5,
    'restaurant-to-laptop': 6,
    'service-to-laptop': 7,
    'restaurant-to-device': 8,
    'service-to-device': 9,

    }


    dataset_tau_dict = {
    'service-to-restaurant': 3,
    'laptop-to-restaurant': 3,
    'device-to-restaurant': 3,
    'restaurant-to-service': 3,
    'laptop-to-service': 3,
    'device-to-service': 3,
    'restaurant-to-laptop': 3,
    'service-to-laptop': 3,
    'restaurant-to-device': 3,
    'service-to-device': 3
    }

    opt.idx = pair_index[pair]
    opt.tau = dataset_tau_dict[pair]

    if not os.path.exists('./log/{}.{}'.format(opt.idx, pair)):
        os.makedirs('./log/{}.{}'.format(opt.idx, pair))

    log_file = './log/{}.{}/{}-{}.log'.format(opt.idx, pair, opt.name, strftime("%y%m%d-%H%M%S", localtime()))

    logger.addHandler(logging.FileHandler(log_file))
    logger.info('> log file: {}'.format(log_file))
    ins = Instructor(opt)
    ins.run()

    end_time = time()
    logger.info('Running Time: {:.0f}m {:.0f}s'.format((end_time-start_time) // 60, (end_time-start_time) % 60))

if __name__ == '__main__':
    main()
