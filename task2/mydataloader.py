#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 11:25 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : mydataloader.py
# @Software: PyCharm

import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_processing import get_data
from torch.nn.utils.rnn import pad_sequence
import torch
import spacy

spacy_en = spacy.load('en_core_web_sm')

class Language:

    def __init__(self, embedding_size=50, lowercase=True):
        self.embedding_size = embedding_size
        self.lowercase = lowercase
        self.word2id = {}
        self.id2word = {}
        self.word2vec = {}
        self.word2id['<unk>'] = 1
        self.word2id['<pad>'] = 0
        self.id2word[0] = '<pad>'
        self.id2word[1] = '<unk>'
        self.word2vec['<unk>'] = [0.0] * embedding_size
        self.word2vec['<pad>'] = [0.0] * embedding_size

    def build_word2vec(self, word_vec_path):
        with open(word_vec_path, 'r', encoding="utf-8") as f:
            sents = f.readlines()
            for i, sent in enumerate(sents, 2):
                if self.lowercase:
                    sent = sent.lower()
                words = sent.strip().split(' ')
                self.word2id[words[0]] = i
                self.id2word[i] = words[0]
                self.word2vec[words[0]] = [float(item) for item in words[1:]]
        self.embedding = np.zeros((len(self.word2vec), self.embedding_size))
        for id, word in self.id2word.items():
            self.embedding[id] = np.array(self.word2vec[word])
        return self.word2id, self.id2word, self.embedding

    def fit(self, sents):
        X = []
        for sent in sents:
            if self.lowercase:
                sent = sent.lower()
            words = sent.strip().split(' ')
            words_id = []
            for word in words:
                if word not in self.word2id:
                    words_id.append(self.word2id['<unk>'])
                else:
                    words_id.append(self.word2id[word])
            X.append(words_id)
        return X




class Mydataset(Dataset):

    def __init__(self, sents, sentiments):
        super(Mydataset, self).__init__()
        self.x = sents
        self.y = sentiments

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def my_collate_fn(batch_data):
    """ 自定义一个batch里面的数据的组织方式 """
    batch_data.sort(key=lambda data_pair: len(data_pair[0]), reverse=True)

    sents, labels = zip(*batch_data)
    sents_len = [len(sent) for sent in sents]
    sents = [torch.LongTensor(sent) for sent in sents]
    padded_sents = pad_sequence(sents, batch_first=True, padding_value=0)

    return torch.LongTensor(padded_sents), torch.LongTensor(labels), torch.FloatTensor(sents_len)


def tokenize(sent):
    return [token.text for token in spacy_en.tokenizer(sent)]






if __name__ == '__main__':
    language = Language()
    train, test = get_data('data')
    word2id, id2word, embedding = language.build_word2vec('./pretrained_wordvector/glove.6B.50d.txt')
    print(embedding.shape)
    X = language.fit(train['Phrase'].values)
    print(type(embedding))
    # train_dataset = Mydataset(X, train['Sentiment'].values)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, collate_fn=my_collate_fn)
    # c = 0
    # for x, y, l in train_dataloader:
    #     c += 1
    #     print(x, y, l, sep='----')
    #     if c > 10:
    #         break


