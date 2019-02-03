#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 01:39:22 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from brown import get_sentences_with_word2idx_limit_vocab
from markov import get_bigram_probs

sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
V = len(word2idx)
print("Vocab Size:",V)

start_idx = word2idx['START']
end_idx = word2idx['END']

bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

D = 100
W1 = np.random.randn(V,D) / np.sqrt(V)
W2 = np.random.randn(D,V) / np.sqrt(D)

losses = []
epochs = 1
lr = 1e-2

def softmax(a):
    a = a - a.max()
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)

W_bigram = np.log(bigram_probs)
bigram_losses = []

for epoch in range(epochs):
        
    random.shuffle(sentences)
    j = 0
    for sentence in sentences:
        sentence = [start_idx] + sentence + [end_idx]
        n = len(sentence)
        inputs = sentence[:n-1]
        targets = sentence[1:]
        
        hidden = np.tanh(W1[inputs])
        predictions = softmax(hidden.dot(W2))

        loss = -np.sum(np.log(predictions[np.arange(n-1),targets])) / (n-1)
        losses.append(loss)
        
        doutput = predictions
        doutput[np.arange(n-1),targets] -= 1
        W2 = W2 - lr * hidden.T.dot(doutput)
        dhidden = doutput.dot(W2.T) * (1 - hidden*hidden)
        
        i = 0
        for w in inputs:
            W1[w] = W1[w] - lr * dhidden[i]
            i += 1
            
        if epoch == 0:
            bigram_predictions = softmax(W_bigram[inputs])
            bigram_loss = -np.sum(np.log(bigram_predictions[np.arange(n-1),targets])) / (n-1)
            
        if j % 100 == 0:
            print("epoch:", epoch, "sentence: %s/%s" % (j, len(sentences)), "loss:", loss)
        j += 1
plt.plot(losses) 
           
            