#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 1:21 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : models.py
# @Software: PyCharm

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class TextRNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_of_class, weights=None, rnn_type='rnn',
                 num_of_layer=1, bidirectional=False, dropout_rate=0.0):
        super(TextRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_zie = embedding_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_of_layer = num_of_layer
        self.bidirectional = bidirectional
        self.num_of_class = num_of_class

        if weights is None:
            self.weights = nn.Embedding(vocab_size, embedding_size)
        else:
            self.weights = nn.Embedding(vocab_size, embedding_size, _weight=weights)

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True,
                              num_layers=self.num_of_layer, bidirectional=self.bidirectional)
        elif rnn_type == 'lstm':
            self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                                num_layers=self.num_of_layer, bidirectional=self.bidirectional)

        self.dropout = nn.Dropout(dropout_rate)
        if self.bidirectional:
            self.hidden2label = nn.Linear(hidden_size * num_of_layer * 2, num_of_class)
        else:
            self.hidden2label = nn.Linear(hidden_size * num_of_layer, num_of_class)

    def forward(self, X, lens):
        batch_size = len(X)
        embed_X = self.weights(X)  # (batch, seq_len, embedding_size)
        packed_embed_X = pack_padded_sequence(embed_X, lens, batch_first=True)

        if self.rnn_type == 'rnn':
            if self.bidirectional:
                h0 = torch.randn(self.num_of_layer * 2, batch_size, self.hidden_size)
            else:
                h0 = torch.randn(self.num_of_layer, batch_size, self.hidden_size)
            _, hn = self.rnn(packed_embed_X)   # (num_of_layer*bidirectional, batch, hidden)
        elif self.rnn_type == 'lstm':
            if self.bidirectional:
                h0, c0 = torch.randn(self.num_of_layer * 2, batch_size, self.hidden_size), \
                         torch.randn(self.num_of_layer * 2, batch_size, self.hidden_size)
            else:
                h0, c0 = torch.randn(self.num_of_layer, batch_size, self.hidden_size), \
                         torch.randn(self.num_of_layer * 2, batch_size, self.hidden_size)
            _, (hn, cn) = self.lstm(packed_embed_X)   # (num_of_layer*bidirectional, batch, hidden)

        hn = hn.view(batch_size, -1)
        logits = self.hidden2label(self.dropout(hn))
        return logits

if __name__ == '__main__':
    X = torch.LongTensor([[1, 2, 3, 32, 0], [2, 5, 11, 0, 0], [21, 1, 0, 0, 0], [32, 0, 0, 0, 0]])
    lens = torch.Tensor([4, 3, 2, 1])
    textrnn = TextRNN(33, 10, 20, 5)
    print(textrnn(X, lens).size())