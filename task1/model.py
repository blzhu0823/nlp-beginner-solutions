#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 9:35 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : model.py
# @Software: PyCharm


import numpy as np


def softmax(x):
    x -= np.max(x, axis=0, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)


class SoftmaxRegression:

    def __init__(self, feature_num, data_size, num_of_class, learning_rate=0.01, regularization=None):

        self.m = feature_num
        self.n = data_size
        self.c = num_of_class
        self.w = np.random.randn(self.c, self.m)
        # self.b = np.zeros((self.c, 1))
        self.learning_rate = learning_rate
        self.regularization = regularization

    def predict(self, x):
        return softmax(self.w.dot(x))

    def train(self, X, Y, epoch=100, print_loss=0, batch_size=1, shuffle=False):
        loss_history = []
        for e in range(epoch):
            loss = 0.0
            index = np.arange(self.n)
            if shuffle:
                np.random.shuffle(index)
            for i in range(0, self.n, batch_size):
                x = X[:, index[i:i + batch_size]]
                y = Y[:, index[i:i + batch_size]]
                preds = self.predict(x)
                gradient_w = np.dot((preds - y), x.T) / x.shape[1]
                self.w -= self.learning_rate * gradient_w
                if self.regularization == 'l1':
                    self.w -= np.sign(w)
                elif self.regularization == 'l2':
                    self.w -= w
                loss -= np.sum(y * np.log(preds))
            loss /= self.n
            loss_history.append(loss)
            if print_loss > 0 and e % print_loss == 0:
                print('epoch {} loss {}'.format(e, loss))
        return loss_history
