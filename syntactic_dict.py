from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

def softmax(probs):
    probs = np.array(probs)
    probs -= np.max(probs, axis=-1, keepdims=True)
    return np.exp(probs) / (np.sum(np.exp(probs), axis=-1, keepdims=True) + 1e-6)

def read_data(fname):
    max_length = 110

    with open('./data/dep.dict', 'r', encoding='utf-8') as f:
        dep_dict = eval(f.read())
    with open('./data/pos.dict', 'r', encoding='utf-8') as f:
        pos_dict = eval(f.read())
    review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
    pos_data = open(fname + r'pos.txt', 'r', encoding='utf-8').readlines()
    graph_data = pickle.load(open(fname + r'dep.graph', 'rb'))

    word_dict = []
    dep_embs = []
    pos_embs = []

    for sen_index, _ in enumerate(review):

        words = review[sen_index].strip().split()
        dep_graph = graph_data[sen_index]
        pos_line = pos_data[sen_index].strip().split()

        length = len(words)

        '''POS & DEP'''

        for i in range(length):
            word = words[i]

            dep_multihot = [0.] * 40
            dep_slice = dep_graph[i]
            for dep in dep_slice:
                if dep != 0:
                    dep_multihot[dep - 1] = 1.

            pos = pos_line[i]
            pos_onehot = [0.] * 45
            pos_indice = pos_dict[pos] - 1
            pos_onehot[pos_indice] = 1.

            dep_multihot = np.array(dep_multihot)
            pos_onehot = np.array(pos_onehot)

            if word not in word_dict:
                word_dict.append(word)
                dep_embs.append(dep_multihot)
                pos_embs.append(pos_onehot)

            else:
                position = word_dict.index(word)
                dep_embs[position] += dep_multihot
                pos_embs[position] += pos_onehot

    pass
    return word_dict, np.array(pos_embs), np.array(dep_embs)


origins = ['restaurant', 'laptop', 'device']
banks = ['restaurant', 'laptop', 'device']
# banks = ['laptop']

processes = ['train', 'test']
splits = [1, 2, 3]

TOPK = 10

with open('./data/word2id.txt', 'r', encoding='utf-8') as f:
    word2id_dict = eval(f.read())
word2vec = np.load('./data/cross_embedding.npy')

def get_word_embedding(words, word2id_dict, word2vec):
    word_embs = []
    for word in words:
        word_id = word2id_dict[word]
        word_emb = word2vec[word_id]
        word_embs.append(word_emb)

    return np.array(word_embs)


for split in splits:
    for process in processes:
        for origin in origins:
            for bank in banks:
                if bank == 'restaurant':
                    index = 0
                elif bank == 'laptop':
                    index = 1
                elif bank == 'device':
                    index = 2

                with open('./data/pmi_dict_split{}.txt'.format(split), 'r', encoding='utf-8') as f:
                    pmi_dict = eval(f.read())

                with open('./data/frq_dict_split{}.txt'.format(split), 'r', encoding='utf-8') as f:
                    frq_dict = eval(f.read())



                print('SPLIT: {}, ORIGIN: {}, PROCESS: {}, BANK: {}'.format(split, origin, process, bank))
                oracle_dict = {}

                origin_path = './data/{}/{}{}/'.format(origin, process, split)
                bank_path = './data/{}/train{}/'.format(bank, split)


                prototype_f = open('{}{}-to-{}-prototype.txt'.format(origin_path, origin, bank), 'w', encoding='utf-8')
                print('WRITETOFILE: {}\n'.format('{}{}-to-{}-prototype.txt'.format(origin_path, origin, bank)))

                origin_word_dict, origin_pos_embs, origin_dep_embs = read_data(origin_path)
                origin_size = len(origin_word_dict)
                origin_pos_embs = (origin_pos_embs > 0).astype("float32")
                origin_dep_embs = (origin_dep_embs > 0).astype("float32")
                origin_wrd_embs = get_word_embedding(origin_word_dict, word2id_dict, word2vec).astype("float32")

                bank_word_dict, bank_pos_embs, bank_dep_embs = read_data(bank_path)
                bank_size = len(bank_word_dict)
                bank_pos_embs = (bank_pos_embs > 0).astype("float32")
                bank_dep_embs = (bank_dep_embs > 0).astype("float32")
                bank_wrd_embs = get_word_embedding(bank_word_dict, word2id_dict, word2vec).astype("float32")

                bank_pmis = []
                bank_frqs = []
                pmi_th = 0.
                frq_th = 5
                # frq_th = 10
                for bank_word in bank_word_dict:
                    bank_pmi = pmi_dict[bank_word][index]
                    bank_frq = frq_dict[bank_word][index]
                    bank_pmis.append(float(bank_pmi > pmi_th))
                    bank_frqs.append(float(bank_frq > frq_th))

                bank_pmis = np.array(bank_pmis).reshape(1, -1)
                bank_frqs = np.array(bank_frqs).reshape(1, -1)

                '''
                如何计算词之间的映射关系
                1.word相似度 * pos相似度 * dep相似度（缺点是信息重复了，pos和dep信息在syntactic增强中已用到）
                2.仅word相似度（缺点是不考虑出现频次，相似的尾部词会被选中作为模板词）
                3.word相似度 + PMI筛选（仅考虑与领域PMI大于0的词了，常用词被忽略）
                4.word相似度 + PMI筛选 + 词频筛选
                '''
                # pos_similarities = np.matmul(origin_pos_embs, bank_pos_embs.transpose(1, 0))
                # dep_similarities = np.matmul(origin_dep_embs, bank_dep_embs.transpose(1, 0))
                pos_similarities = (cosine_similarity(origin_pos_embs, bank_pos_embs))
                dep_similarities = (cosine_similarity(origin_dep_embs, bank_dep_embs))
                wrd_similarities = (cosine_similarity(origin_wrd_embs, bank_wrd_embs))

                pmi_indicator = bank_pmis.repeat(origin_size, 0)
                frq_indicator = bank_frqs.repeat(origin_size, 0)

                indicator = ((pmi_indicator + frq_indicator) > 0.).astype('float32')

                # similarities = pos_similarities * dep_similarities * wrd_similarities
                # similarities = wrd_similarities
                # similarities = wrd_similarities * indicator
                # similarities = wrd_similarities * indicator * dep_similarities * pos_similarities
                # similarities = (wrd_similarities * dep_similarities * pos_similarities) * indicator
                
                # 12.22 (1)
                # similarities = (wrd_similarities + dep_similarities + pos_similarities) * indicator
                # 12.22 (2)
                similarities = (wrd_similarities * dep_similarities * pos_similarities) * frq_indicator
                # 12.22 (3)
                # similarities = (wrd_similarities + dep_similarities + pos_similarities)
                # 12.22 (4)
                # similarities = (wrd_similarities * dep_similarities * pos_similarities)

                # topk_indices = np.flipud(np.argsort(similarities, axis=-1))[:, :10]

                for origin_index, origin_word in enumerate(origin_word_dict):
                    # topk_index = topk_indices[origin_index]
                    oracle_info = ''
                    topk_index = np.flipud(np.argsort(similarities[origin_index, :]))[:15]
                    topk_words = []
                    topk_similarities = []
                    for k_index in topk_index:
                        if len(topk_words) > TOPK:
                            break

                        k_word = bank_word_dict[k_index]
                        k_similarity = similarities[origin_index, k_index]

                        # if k_word == origin_word:
                        #     continue

                        topk_words.append(k_word)
                        topk_similarities.append(k_similarity)

                    # topk_similarities = softmax(topk_similarities) #保留真实值，在utils里使用softmax

                    for i in range(len(topk_words)):
                        oracle_info += '{} {:.2f}'.format(topk_words[i], topk_similarities[i])
                        if i != len(topk_words) - 1:
                            oracle_info += '@@@'

                    oracle_dict[origin_word] = oracle_info



                    # print('\n')
                    # print(origin_word)
                    # print(' '.join(topk_words))
                    # print(' '.join(topk_similarities))


                sentences = open('{}sentence.txt'.format(origin_path), 'r', encoding='utf-8').readlines()
                for sentence in sentences:
                    words = sentence.strip().split()
                    for word_idx, word in enumerate(words):
                        oracle = oracle_dict[word]
                        prototype_f.write('{}@@@'.format(word_idx))
                        prototype_f.write(oracle)

                        if word_idx != len(words)-1:
                            prototype_f.write('###')
                        else:
                            prototype_f.write('\n')







                pass






