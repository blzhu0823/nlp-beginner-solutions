#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/6 2:11 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : utils.py
# @Software: PyCharm


import torchtext
import os
from torchtext.data import BucketIterator, Iterator
from torchtext import data


def get_iter(datapath, batch_size, vector=None, use_tree=False, device='cpu'):
    if not use_tree:
        TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True, lower=True, include_lengths=True)
        LABEL = data.Field(batch_first=True, sequential=False)
        TREE = None
        fields = {'sentence1': ('premise', TEXT),
                  'sentence2': ('hypothesis', TEXT),
                  'gold_label': ('label', LABEL)}
    else:
        pass

    train_dataset, dev_dataset, test_dataset = data.TabularDataset.splits(
        path=datapath,
        train='snli_1.0_train.jsonl',
        validation='snli_1.0_dev.jsonl',
        test='snli_1.0_test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda x: x.label != '-'
    )
    if vector is not None:
        TEXT.build_vocab(train_dataset, vectors=vector)
    else:
        TEXT.build_vocab(train_dataset)
    LABEL.build_vocab(train_dataset)

    train_iter, dev_iter = BucketIterator.splits(
        (train_dataset, dev_dataset),
        batch_size=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.premise) + len(x.hypothesis),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )
    test_iter = Iterator(
        test_dataset,
        batch_size=batch_size,
        device=device,
        sort=False,
        sort_within_batch=False,
        repeat=False,
        shuffle=False
    )
    return train_iter, dev_iter, test_iter, TEXT, LABEL, TREE

if __name__ == '__main__':
    # i = 0
    # with open("data/snli_1.0/snli_1.0_test.jsonl", "r+", encoding="utf8") as f:
    #     for x in f:
    #         print(x)
    #         i += 1
    #         if i > 1000:
    #             break
    get_iter('data/snli_1.0', 32)
