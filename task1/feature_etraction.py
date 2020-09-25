#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 2:17 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : feature_etraction.py
# @Software: PyCharm


import data_processing
import numpy as np
from collections import Counter


class Ngram:

    def __init__(self, ngram, do_lowercase=True):
        self.ngram = ngram
        self.do_lowercase = do_lowercase
        self.vocab2id = {}

    def build_vocab(self, sents: list) -> object:
        for sent in sents:
            if self.do_lowercase:
                sent = sent.lower()
            words = sent.strip().split(' ')
            for gram in self.ngram:
                for i in range(len(words) - gram + 1):
                    ngram_word = '|'.join(words[i:i + gram])
                    if ngram_word not in self.vocab2id:
                        self.vocab2id[ngram_word] = len(self.vocab2id)

    def fit(self, sents: list) -> np.ndarray:
        X = np.zeros((len(sents), len(self.vocab2id)))
        for row, sent in enumerate(sents):
            if self.do_lowercase:
                sent = sent.lower()
            words = sent.strip().split(' ')
            for gram in self.ngram:
                for i in range(len(words) - gram + 1):
                    ngram_word = '|'.join(words[i:i + gram])
                    col = self.vocab2id[ngram_word]
                    X[row][col] += 1
        return X.T

def transform_to_onehot(y, num_of_class):
    result = np.zeros((num_of_class, len(y)))
    for i, pos in enumerate(y):
        result[pos][i] = 1
    return result


if __name__ == '__main__':
    sentences = ['I love you Jack .', 'I love you too Mary .']
    ngram_model = Ngram([1, 2])
    ngram_model.build_vocab(sentences)
    result = ngram_model.fit(sentences)
    print(result.shape)
