#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 3:01 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : model.py
# @Software: PyCharm


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, layernum=1, dropout_rate=0.3):
        super(LSTM, self).__init__()
        if layernum > 1:
            self.lstm = nn.LSTM(input_size, hidden_size, layernum, dropout=dropout_rate, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, layernum, batch_first=True)
        self.layernum = layernum
        self.input_size = input_size

    def forward(self, x, lens, hidden):
        """

        :param hidden: a tuple, both item with shape (layernum, batch, hidden_size)
        :param lens: (batch, )
        :param x: (batch, len, input_size)
        :return: out: (batch, len, hidden_size)
                 (h, c): a tuple, both item with shape (layernum, batch, hidden_size)
        """
        batch_size = len(lens)
        packed_x = pack_padded_sequence(x, lens, batch_first=True)
        packed_out, (h, c) = self.lstm(packed_x, hidden)
        out, _= pad_packed_sequence(packed_out, batch_first=True)
        return out, (h, c)


class LM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size=128, layernum=1, dropout_rate=0.3):
        super(LM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(embedding_size, hidden_size, layernum, dropout_rate)
        self.project = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernum = layernum
        self.hidden_size = hidden_size

    def forward(self, x, lens, hidden=None):
        """

        :param hidden: a tuple, both item with shape (layernum, batch, hidden_size)
        :param x: (batch, len)
        :param lens: (batch, )
        :return:
        """
        batch_size = len(lens)
        if hidden is None:
            hidden = torch.zeros(self.layernum, batch_size, self.hidden_size), \
                     torch.zeros(self.layernum, batch_size, self.hidden_size)
        embed_x = self.embed(x) # (batch, len, input_size)
        out, (hn, cn) = self.lstm(self.dropout(embed_x), lens, hidden)
        logits = self.project(self.dropout(out))  # (batch, len, vocab_size)
        return logits, (hn, cn)


if __name__ == '__main__':
    # a test
    lm = LM(10, 20)
    x = torch.LongTensor([[1, 4, 3, 0, 6, 5], [3, 4, 4, 2, 2, 2]])
    lens = torch.LongTensor([6, 3])
    lm(x, lens)

