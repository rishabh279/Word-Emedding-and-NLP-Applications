#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 00:45:35 2018

@author: rishabh
"""
import numpy as np
import matplotlib.pyplot as plt 
import random
from brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

from markov import get_bigram_probs

sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)

V = len(word2idx)
print("Vocab size:",V)

start_idx = word2idx['START']
end_idx = word2idx['END']

bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing= 0.1)

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
        inputs = np.zeros((n-1,V))
        targets = np.zeros((n-1,V))
        inputs[np.arange(n-1),sentence[:n-1]] = 1
        targets[np.arange(n-1),sentence[1:]] = 1
        
        hidden = np.tanh(inputs.dot(W1))
        predictions = softmax(hidden.dot(W2))
        
        W2 = W2 - lr * hidden.T.dot(predictions - targets)
        dhidden = (predictions - targets).dot(W2.T) * (1 - hidden * hidden)
        W1 = W1 - lr * inputs.T.dot(dhidden)
        
        loss = -np.sum(targets * np.log(predictions)) / (n-1)
        losses.append(loss)
        
        if epoch == 0:
            bigram_predictions = softmax(inputs.dot(W_bigram))
            bigram_loss = -np.sum(targets * np.log(bigram_predictions)) / (n-1)
            bigram_losses.append(bigram_loss)
            
        if j % 10 == 0:
            print("epoch:", epoch, "sentence: %s/%s" % (j,len(sentences)), "loss:", loss)
        j = j + 1
        
plt.plot(losses)

avg_bigram_loss = np.mean(bigram_losses)
print("avg_bigram_loss:", avg_bigram_loss)
plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')

    