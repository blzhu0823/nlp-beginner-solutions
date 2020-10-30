#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 4:39 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm

import math
import torch
import torch.nn as nn
from model import LM
from utils import load_iters
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange




EMBEDDING_SIZE = 300
HIDDENSIZE = 512
LAYERNUM = 1
DROPOUT = 0.3
LEARNING_RATE = 0.01
EPOCHS = 30
CLIP = 5


def train(train_iter, dev_iter, loss_func, optimizer, epochs=10, clip=5):
    for e in trange(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_iter:
            optimizer.zero_grad()
            x, lens = batch.text   # x: (batch, max_len)  lens: (batch, )
            inputs = x[:, :-1]
            target = x[:, 1:]  # (batch, len)
            logits, _ = model(inputs, lens-1)  # (batch, len, vocab_size)
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
        tqdm.write('Epoch {}, Train perplexity: {}'.format(e+1, math.exp(total_loss)))
        writer.add_scalar('Train_loss', total_loss, e+1)

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for batch in dev_iter:
                x, lens = batch.text
                inputs = x[:, :-1]
                target = x[:, 1:]
                logits, _ = model(inputs, lens - 1)
                loss = loss_func(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
                total_loss += loss.item()
            tqdm.write('Epoch {}, Dev perplexity: {}'.format(e+1, math.exp(total_loss)))
            writer.add_scalar('Dev_loss', total_loss, e+1)

def eval(test_iter, loss_func):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for batch in test_iter:
            x, lens = batch.text
            inputs = x[:, :-1]
            target = x[:, 1:]
            logits, _ = model(inputs, lens - 1)
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            total_loss += loss.item()
        tqdm.write('{} perplexity: {}'.format('Test', math.exp(total_loss)))


def generate(end_token, words, word2id, id2word, model, strategy='greedy'):
    wordlist = ['<start>']
    wordlist.extend(words)
    input = [word2id[word] for word in wordlist]
    input = torch.LongTensor(input).unsqueeze(0)
    lens = input.shape[-1]
    model.eval()
    with torch.no_grad():
        out, (hn, cn) = model(input, torch.LongTensor([lens]))
        if strategy == 'greedy':
            idx = out.squeeze()[-1].argmax()
            while idx != end_token:
                wordlist.append(id2word[idx])
                input = torch.LongTensor([idx]).unsqueeze(0)
                out, (hn, cn) = model(input, torch.LongTensor([1]), (hn, cn))
                idx = out.squeeze().argmax()
        elif strategy == 'random':
            pass
    return ''.join(wordlist[1:])






if __name__ == '__main__':
    train_iter, dev_iter, test_iter, TEXT = load_iters()
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    end_idx = TEXT.vocab.stoi[TEXT.eos_token]
    model = LM(len(TEXT.vocab), EMBEDDING_SIZE, HIDDENSIZE, LAYERNUM, DROPOUT)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = CrossEntropyLoss(ignore_index=pad_idx)
    writer = SummaryWriter('logs')
    train(train_iter, dev_iter, loss_function, optimizer, EPOCHS, CLIP)
    eval(test_iter, loss_function)
    text = generate(end_idx, '山河', TEXT.vocab.stoi, TEXT.vocab.itos, model)
    print(text)