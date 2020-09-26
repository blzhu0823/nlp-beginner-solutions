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
from models import TextRNN, TextCNN
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    epoch_num = 10
    learning_rate = 0.001
    num_of_class = 5
    model_type = 'cnn'  # ['rnn', 'lstm', 'cnn']

    train, test = get_data('data')
    language = Language()
    word2id, id2word, embedding = language.build_word2vec('./pretrained_wordvector/glove.6B.50d.txt')

    train_sents, val_sents, train_labels, val_labels = train_test_split(train['Phrase'].values,
                                                                        train['Sentiment'].values,
                                                                        test_size=0.3,
                                                                        random_state=0,
                                                                        stratify=train['Sentiment'].values)


    train_dataset = Mydataset(language.fit(train_sents), train_labels)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128,
                                  shuffle=True, collate_fn=my_collate_fn)
    val_dataset = Mydataset(language.fit(val_sents), val_labels)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=128,
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
        model = TextCNN(vocab_size=len(word2id), embedding_size=50,
                        num_of_class=5, weights=torch.Tensor(embedding))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = CrossEntropyLoss()

    for e in range(epoch_num):

        model.eval()
        val_accs = []
        for X, y, lens in val_dataloader:
            if model_type in ['rnn', 'lstm']:
                logits = model(X, lens)
            elif model_type is 'cnn':
                logits = model(X)
            preds = torch.argmax(logits, dim=1)
            acc = torch.mean((preds == y).float())
            val_accs.append(acc)
        val_acc = sum(val_accs)/len(val_accs)
        print('epoch {} train acc: {}%'.format(e, val_acc))


        model.train()
        for X, y, lens in train_dataloader:
            if model_type in ['rnn', 'lstm']:
                logits = model(X, lens)
            elif model_type is 'cnn':
                logits = model(X)
            optimizer.zero_grad()
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
