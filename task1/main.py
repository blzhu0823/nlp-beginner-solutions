#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 3:44 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm


import numpy as np
from feature_etraction import transform_to_onehot, Ngram
from data_processing import get_data
from model import SoftmaxRegression

train, test = get_data('data')
# print(train.columns)
ngram = Ngram([1])
ngram.build_vocab(train['Phrase'].values)
train_x = ngram.fit(train['Phrase'].values)
train_y = transform_to_onehot(train['Sentiment'].values, 5)
print(train_x.shape)
model = SoftmaxRegression(feature_num=train_x.shape[0], data_size=train_x.shape[1],
                          num_of_class=train_y.shape[0])
model.train(X=train_x, Y=train_y, batch_size=128, shuffle=True, print_loss=1, epoch=10)
