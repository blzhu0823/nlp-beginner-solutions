#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 12:55 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : utils.py
# @Software: PyCharm


import os
import torchtext.data as data
from torchtext.data import BucketIterator

def read_data(input_file):
    poetries = []
    poetry = []
    with open(input_file, encoding='utf-8') as f:
        text = f.read()
    lines = text.replace('\n', '').strip('。').split('。')
    lines = list(map(lambda x:x + '。', lines))
    for line in lines:
        poetry.extend(line)
        poetries.append(poetry)
        poetry = []
    return poetries


class PoetryDataset(data.Dataset):
    def __init__(self, datafile, text_field, **kwargs):
        examples = []
        fields = [('text', text_field)]
        poetries = read_data(datafile)
        for line in poetries:
            examples.append(data.Example.fromlist([line], fields))
        super(PoetryDataset, self).__init__(examples, fields, **kwargs)



def load_iters(data_file='./poetryFromTang.txt', batch_size=32, device='cpu'):
    TEXT = data.Field(sequential=True, eos_token='<end>', init_token='<start>', batch_first=True, include_lengths=True)
    poetrydataset = PoetryDataset(datafile=data_file, text_field=TEXT)
    train, dev, test = poetrydataset.split([0.8, 0.1, 0.1])
    TEXT.build_vocab(train)
    train_iter, dev_iter, test_iter = BucketIterator.splits((train, dev, test),
                                                            batch_sizes=(batch_size, batch_size, batch_size),
                                                            shuffle=True, repeat=False,
                                                            sort_key=lambda x: len(x.text),
                                                            sort_within_batch=True,
                                                            device=device)
    return train_iter, dev_iter, test_iter, TEXT






if __name__ == '__main__':
    train_iter, dev_iter, test_iter, TEXT = load_iters()
