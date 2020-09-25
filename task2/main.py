#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 7:42 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from data_processing import get_data
from mydataloader import Language, Mydataset, my_collate_fn
from models import TextRNN

if __name__ == '__main__':

    epoch_num = 10
    learning_rate = 0.001
    num_of_class = 5
    model_type = 'rnn'  # ['rnn', 'lstm', 'cnn']

    train, test = get_data('data')
    language = Language()
    word2id, id2word, embedding = language.build_word2vec('./pretrained_wordvector/glove.6B.50d.txt')
    sents = language.fit(train['Phrase'].values)
    train_dataset = Mydataset(sents, train['Sentiment'].values)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128,
                                  shuffle=True, collate_fn=my_collate_fn)
    if model_type == 'rnn':
        model = TextRNN(vocab_size=len(word2id), embedding_size=50,
                        hidden_size=50, num_of_class=5,
                        weights=torch.Tensor(embedding), rnn_type='rnn')
    elif model_type == 'lstm':
        model = TextRNN(vocab_size=len(word2id), embedding_size=50,
                        hidden_size=50, num_of_class=5,
                        weights=torch.Tensor(embedding), rnn_type='lstm')
    elif model_type == 'cnn':
        pass
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = CrossEntropyLoss()

    for e in range(epoch_num):

        model.eval()
        train_accs = []
        for X, y, lens in train_dataloader:
            logits = model(X, lens)
            preds = torch.argmax(logits, dim=1)
            acc = torch.mean(torch.tensor(preds == y, dtype=torch.float))
            train_accs.append(acc)
        train_acc = sum(train_accs)/len(train_accs)
        print('epoch {} train acc: {}%'.format(e, train_acc))


        model.train()
        for X, y, lens in train_dataloader:
            logits = model(X, lens)
            optimizer.zero_grad()
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
