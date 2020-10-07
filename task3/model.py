#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/7 1:23 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : model.py
# @Software: PyCharm

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_of_layers=1, dropout_rate=0.1):
        super(BiLSTM, self).__init__()
        if num_of_layers > 1:
            self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2,
                                  bidirectional=True, num_layers=num_of_layers,
                                  dropout=dropout_rate, batch_first=True)
        else:
            self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2,
                                  bidirectional=True, num_layers=num_of_layers,
                                  batch_first=True)

    def forward(self, x, lens):
        """

        :param x: (batch, seq_len, input_size)
        :param lens: (batch,)
        """
        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]
        packed_x = pack_padded_sequence(ordered_x, ordered_lens, batch_first=True)
        packed_out, _ = self.bilstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq_len, hidden_size)
        recovered_index = index.argsort()
        recovered_out = out[recovered_index]  # (batch, seq_len, hidden_size)
        return recovered_out


class ESIM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_of_class,
                 dropout_rate=0.1, num_of_layers=1, pretrained_weight=None, freeze=False):
        super(ESIM, self).__init__()
        if pretrained_weight is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_weight, freeze=freeze)
        else:
            self.embed = nn.Embedding(vocab_size, embedding_size)
        self.bilstm1 = BiLSTM(embedding_size, hidden_size, num_of_layers=num_of_layers,
                              dropout_rate=dropout_rate)
        self.bilstm2 = BiLSTM(hidden_size, hidden_size, num_of_layers=num_of_layers,
                              dropout_rate=dropout_rate)
        self.fc1 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_of_class)
        self.dropout = nn.Dropout(dropout_rate)

    def soft_align_attention(self, x1, x1_lens, x2, x2_lens):
        """

        :param x1: (batch, seq1_len, hidden_size)
        :param x1_lens: (batch,)
        :param x2: (batch, seq2_len, hidden_size)
        :param x2_lens: (batch,)
        """

        batch, seq1_len, seq2_len = x1.size(0), x1.size(1), x2.size(1)
        attention = torch.matmul(x1, x2.transpose(1, 2))  # (batch, seq1_len, seq2_len)
        mask1 = torch.arange(seq1_len).expand(batch, seq1_len) >= x1_lens.unsqueeze(1)
        mask2 = torch.arange(seq2_len).expand(batch, seq2_len) >= x2_lens.unsqueeze(1)
        mask1 = mask1.float().masked_fill(mask1, float('-inf'))  # (batch, seq1_len)
        mask2 = mask2.float().masked_fill(mask2, float('-inf'))  # (batch, seq2_len)
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)  # (batch, seq1_len, seq2_len)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)  # (batch, seq2_len, seq1_len)
        x1_align = torch.matmul(weight1, x2)
        x2_align = torch.matmul(weight2, x1)
        return x1_align, x2_align

    def composition(self, x1, x1_lens, x2, x2_lens):
        """

        :param x1: (batch, seq1_len, 4*hidden_size)
        :param x1_lens: (batch,)
        :param x2: (batch, seq2_len, 4*hidden_size)
        :param x2_lens: (batch,)
        """

        x1 = F.relu(self.fc1(x1))  # (batch, seq1_len, hidden_size)
        x2 = F.relu(self.fc1(x2))  # (batch, seq2_len, hidden_size)
        x1_compose = self.bilstm2(self.dropout(x1), x1_lens)  # (batch, seq1_len, hidden_size)
        x2_compose = self.bilstm2(self.dropout(x2), x2_lens)  # (batch, seq2_len, hidden_size)
        x1_max = F.max_pool1d(x1_compose.transpose(1, 2), x1_compose.size(1)).squeeze(-1)  # (batch, hidden_size)
        x1_avg = F.avg_pool1d(x1_compose.transpose(1, 2), x1_compose.size(1)).squeeze(-1)  # (batch, hidden_size)
        x2_max = F.max_pool1d(x2_compose.transpose(1, 2), x2_compose.size(1)).squeeze(-1)  # (batch, hidden_size)
        x2_avg = F.avg_pool1d(x2_compose.transpose(1, 2), x2_compose.size(1)).squeeze(-1)  # (batch, hidden_size)
        return torch.cat([x1_max, x1_avg, x2_max, x2_avg], dim=-1)  # (batch, 4*hidden_size)

    def forward(self, x1, x1_lens, x2, x2_lens):
        """

        :param x1: (batch, seq1_len)
        :param x1_lens: (batch,)
        :param x2: (batch, seq2_len)
        :param x2_lens: (batch,)
        """
        embed_x1 = self.embed(x1)  # (batch, seq1_len, embedding_size)
        embed_x2 = self.embed(x2)  # (batch, seq2_len, embedding_size)
        encode_x1 = self.bilstm1(self.dropout(embed_x1), x1_lens)  # (batch, seq1_len, hidden_size)
        encode_x2 = self.bilstm1(self.dropout(embed_x2), x2_lens)  # (batch, seq1_len, hidden_size)
        x1_align, x2_align = self.soft_align_attention(encode_x1, x1_lens, encode_x2, x2_lens)

        x1_enhanced = torch.cat([encode_x1, x1_align, encode_x1 - x1_align, encode_x1 * x1_align],
                                dim=-1)  # (batch, seq1_len, 4*hidden_size)
        x2_enhanced = torch.cat([encode_x2, x2_align, encode_x2 - x2_align, encode_x2 * x2_align],
                                dim=-1)  # (batch, seq1_len, 4*hidden_size)
        composed = self.composition(x1_enhanced, x1_lens, x2_enhanced, x2_lens)  # (batch, 4*hidden_size)
        out = self.fc3(self.dropout(torch.tanh(self.fc2(self.dropout(composed)))))
        return out


if __name__ == '__main__':
    model = ESIM(10, 20, 50, 5)
    x1 = torch.tensor([[6, 2, 3, 3, 6, 8, 1, 1, 1], [2, 3, 4, 2, 3, 4, 5, 3, 2]])
    x1_lens = torch.tensor([6, 9])
    x2 = torch.tensor([[4, 3, 2, 3, 2, 1, 1], [2, 3, 1, 2, 3, 0, 0]])
    x2_lens = torch.tensor([5, 7])
    print(x1.shape, x1_lens.shape, x2.shape, x2_lens.shape)
    out = model(x1, x1_lens, x2, x2_lens)
    print(out.shape, out)
