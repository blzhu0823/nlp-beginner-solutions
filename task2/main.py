#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 7:42 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torchtext
import pandas as pd
import os
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from data_processing import get_data
from mydataloader import Language, Mydataset, my_collate_fn, tokenize
from models import TextRNN, TextCNN
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    epoch_num = 10
    learning_rate = 0.001
    num_of_class = 5
    model_type = 'lstm'  # ['rnn', 'lstm', 'cnn']
    load_data_by_torchtext = True  # if True, use torchtext to load data, else load data by hand

    train, test = get_data('data')
    train_sents, val_sents, train_labels, val_labels = train_test_split(train['Phrase'].values,
                                                                        train['Sentiment'].values,
                                                                        test_size=0.3,
                                                                        random_state=0,
                                                                        stratify=train['Sentiment'].values)

    if load_data_by_torchtext:
        train_df = pd.DataFrame({'text': train_sents, 'label': train_labels})
        val_df = pd.DataFrame({'text': val_sents, 'label': val_labels})
        train_path = os.path.join('data', 'train.csv')
        val_path = os.path.join('data', 'val.csv')
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        TEXT = torchtext.data.Field(sequential=True, use_vocab=True,
                                    lower=True, tokenize=tokenize,
                                    include_lengths=True, batch_first=True)
        LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
        train_dataset, val_dataset = torchtext.data.TabularDataset.splits(path='', train=train_path,
                                                                          validation=val_path, format='csv',
                                                                          skip_header=True,
                                                                          fields=[('text', TEXT), ('label', LABEL)])
        TEXT.build_vocab(train_dataset, vectors='glove.6B.50d')
        train_dataloader = torchtext.data.BucketIterator(train_dataset, batch_size=128, sort_key=lambda x: len(x.text),
                                                         device='cpu', sort_within_batch=True)
        val_dataloader = torchtext.data.BucketIterator(val_dataset, batch_size=128, sort_key=lambda x: len(x.text),
                                                       device='cpu', sort_within_batch=True)
        vocab_size = len(TEXT.vocab.vectors)
        embedding = TEXT.vocab.vectors


    else:
        language = Language()
        word2id, id2word, embedding = language.build_word2vec('./pretrained_wordvector/glove.6B.50d.txt')
        train_dataset = Mydataset(language.fit(train_sents), train_labels)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=128,
                                      shuffle=True, collate_fn=my_collate_fn)
        val_dataset = Mydataset(language.fit(val_sents), val_labels)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=128,
                                    shuffle=True, collate_fn=my_collate_fn)
        vocab_size = len(word2id)
        embedding = torch.Tensor(embedding)

    if model_type == 'rnn':
        model = TextRNN(vocab_size=vocab_size, embedding_size=50,
                        hidden_size=50, num_of_class=5,
                        weights=embedding, rnn_type='rnn')
    elif model_type == 'lstm':
        model = TextRNN(vocab_size=vocab_size, embedding_size=50,
                        hidden_size=50, num_of_class=5,
                        weights=embedding, rnn_type='lstm')
    elif model_type == 'cnn':
        model = TextCNN(vocab_size=vocab_size, embedding_size=50,
                        num_of_class=5, weights=embedding)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = CrossEntropyLoss()

    for e in range(epoch_num):

        model.eval()
        val_accs = []
        for batch in val_dataloader:
            if load_data_by_torchtext:
                (X, lens), y = batch.text, batch.label
            else:
                X, y, lens = batch
            if model_type in ['rnn', 'lstm']:
                logits = model(X, lens)
            elif model_type is 'cnn':
                logits = model(X)
            preds = torch.argmax(logits, dim=1)
            acc = torch.mean((preds == y).float())
            val_accs.append(acc)
        val_acc = sum(val_accs) / len(val_accs)
        print('epoch {} val acc: {}%'.format(e, val_acc))

        model.train()
        for batch in val_dataloader:
            if load_data_by_torchtext:
                (X, lens), y = batch.text, batch.label
            else:
                X, y, lens = batch
            if model_type in ['rnn', 'lstm']:
                logits = model(X, lens)
            elif model_type is 'cnn':
                logits = model(X)
            optimizer.zero_grad()
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
