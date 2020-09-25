#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 12:45 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : data_processing.py
# @Software: PyCharm

import pandas as pd
import os


def get_data(path):
    """

    :param path: tsv data's path
    :return: data
    """
    train_df = pd.read_csv(os.path.join(path, 'train.tsv'), sep='\t')
    test_df = pd.read_csv(os.path.join(path, 'test.tsv'), sep='\t')
    return train_df, test_df


if __name__ == '__main__':
    train, test = get_data('data')
    print(len(train), len(test))
    print(train.columns)
    print(train['Sentiment'].values)