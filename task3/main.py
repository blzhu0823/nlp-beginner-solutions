#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/7 4:01 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm


from utils import get_iter
from tqdm import tqdm
from model import ESIM
import torch.nn as nn
import torch.optim as optim
import torchtext
import torch

BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_TREE = False
HIDDEN_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 1e-3
CLIP = 5
EMBEDDING_SIZE = 300
DROPOUT_RATE = 0.5
LAYERS_NUM = 1
PATIENCE = 5
FREEZE = False
DATA_PATH = 'data/snli_1.0'


def eval(data_iter, name, epoch=None, use_cache=False):
    if use_cache:
        model.load_state_dict(torch.load('checkpoint/best_model.pkl'))
    model.eval()
    correct = 0
    error = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_iter:
            x1, x1_lens = batch.premise
            x2, x2_lens = batch.hypothesis
            y = batch.label

            output = model(x1, x1_lens, x2, x2_lens)  # (batch, num_of_class)
            pred = output.argmax(dim=-1)
            total_loss += loss_fun(output, y).item()
            correct += (pred == y).float().sum().item()
            error += (pred != y).float().sum().item()

    acc = correct / (correct + error)
    if epoch is not None:
        tqdm.write("Epoch %s, %s acc: %.3f, Loss: %.3f" % (epoch, name, acc, total_loss))
    else:
        tqdm.write('%s acc: %.3f, Loss: %.3f' % (name, acc, total_loss))
    return acc

def train(train_iter, dev_iter, epochs, patience=5, clip=5):
    best_acc = -1.0
    patience_counter = 0
    for e in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_iter):
            x1, x1_lens = batch.premise
            x2, x2_lens = batch.hypothesis
            y = batch.label
            model.zero_grad()
            output = model(x1, x1_lens, x2, x2_lens)  # (batch, num_of_class)
            loss = loss_fun(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            total_loss += loss
            optimizer.step()
        tqdm.write("Epoch %s, train loss: %.3f" % (e, total_loss))

        acc = eval(dev_iter, 'dev', e, use_cache=False)
        if acc <= best_acc:
            patience_counter += 1
        else:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoint/best_model.pkl')
        if patience_counter > patience:
            tqdm.write('Early stop: patience limit reached at epoch %s' % e)
            break



if __name__ == '__main__':
    vectors = torchtext.vocab.Vectors('glove.840B.300d.txt', '../pretrained_vectors')
    train_iter, dev_iter, test_iter, TEXT, LABEL, TREE = get_iter(DATA_PATH, BATCH_SIZE, vectors, USE_TREE, DEVICE)
    model = ESIM(vocab_size=len(TEXT.vocab), embedding_size=EMBEDDING_SIZE,
                 hidden_size=HIDDEN_SIZE, num_of_class=len(LABEL.vocab),
                 dropout_rate=DROPOUT_RATE, num_of_layers=LAYERS_NUM,
                 pretrained_weight=TEXT.vocab.vectors, freeze=FREEZE)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(train_iter, dev_iter, EPOCHS, PATIENCE, CLIP)
    eval(test_iter, 'test', use_cache=True)