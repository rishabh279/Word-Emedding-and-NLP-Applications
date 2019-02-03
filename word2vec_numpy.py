#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 02:02:20 2018

@author: rishabh
"""

import json 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import expit as sigmoid

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances


import os 
import sys
import string 

from brown import get_sentences_with_word2idx_limit_vocab as get_brown 

def train_model(savedir):
    # get the data
    sentences, word2idx = get_brown()
    # number of unique words
    vocab_size = len(word2idx)
    # config
    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    #num_negative = 5
    epochs = 20
    D = 50
    
    # learning rate decay
    learning_rate_delta = (learning_rate- final_learning_rate) / epochs
    
    #params
    W = np.random.randn(vocab_size, D)
    V = np.random.randn(D, vocab_size)
    
    # distribution for drawing negative sam
    p_neg = get_negative_sampling_distribution(sentences, vocab_size)
    
    # save the costs to plot them per iteration
    costs = []
    
    # number of total words in corpus
    total_words = sum(len(sentence) for sentence in sentences)
    print("Total number of words in corpus", total_words)

    # for subsampling each sentence
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)
    
    # train the model
    for epoch in range(epochs):
        # randomly order sentences so we don't always see
        # sentences in the same order
        np.random.shuffle(sentences)
        cost = 0
        counter = 0
        for sentence in sentences:
            # keep only certain words based on p_neg
            sentence = [w for w in sentence if np.random.random() < (1 - p_drop[w])]
            if len(sentence) < 2:
                continue
            # randomly order words so we don't always see
            # samples in the same order
            randomly_ordered_positions = np.random.choice(
                    len(sentence),
                    size=len(sentence),
                    replace=False
                    )    
            for pos in randomly_ordered_positions:
                # the middle word
                word = sentence[pos]
                
                # get the positive context words/negative samples                
                context_words = get_context(pos, sentence, window_size)
                neg_words = np.random.choice(vocab_size, p=p_neg)
                targets = np.array(context_words)
                print(neg_words)
                
                # do one iteration of stochastic gradient descent
                c = sgd(word, targets, 1, learning_rate, W, V)
                cost += c
                c = sgd(neg_words, targets, 0, learning_rate, W, V)
                cost += c
            counter += 1
            if counter % 100 == 0:
                sys.stdout.write("processed %s / %s\r"%(counter, len(sentences)))
                sys.stdout.flush()
        
        # save the cost        
        costs.append(cost)
        
        # update the learning rate
        learning_rate -= learning_rate_delta
        
    # plot the cost per iteration        
    plt.plot(costs)
    plt.show()
    
    # save the model
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open('%s/word2idx.json' % savedir, 'w') as f:
         json.dump(word2idx, f)

    np.savez('%s/weights.npz' % savedir, W, V)                
    
    # return the model
    return word2idx, W, V

def get_context(pos, sentence, window_size):
      # input:
      # a sentence of the form: x x x x c c c pos c c c x x x x
      # output:
      # the context word indices: c c c c c c
    start = max(0, pos-window_size)
    end = min(len(sentence), pos + window_size)    
    
    context = []
    for ctx_pos, ctx_word_idx in enumerate(sentence[start:end], start=start):
        if ctx_pos !=pos:
            # don't include the input word itself as a target
            context.append(ctx_word_idx)
            
    return context            
    
def sgd(input_, targets, label, learning_rate, W, V):
      # W[input_] shape: D
      # V[:,targets] shape: D x N
      # activation shape: N
      # print("input_:", input_, "targets:", targets)
    activation = W[input_].dot(V[:,targets])
    prob = sigmoid(activation)

    # gradients
    gV = np.outer(W[input_],prob - label)
    gW = np.sum((prob - label)*V[:,targets],axis=1)

    V[:,targets] -= learning_rate*gV
    W[input_] -= learning_rate*gW
    # return cost (binary cross entropy)
    cost = label * np.log(prob + 1e-10) + (1-label) * np.log(1 - prob + 1e-10)
    return cost.sum()
            
def get_negative_sampling_distribution(sentences, vocab_size):
      # Pn(w) = prob of word occuring
      # we would like to sample the negative samples
      # such that words that occur more often
      # should be sampled more often
    word_freq = np.zeros(vocab_size)
    #word_count = sum(len(sentence) for sentence in sentences)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
    
    # smooth it
    p_neg = word_freq**0.75
    
    # normalize it
    p_neg = p_neg / p_neg.sum()        
    
    return p_neg

def load_model(savedir):
  with open('%s/word2idx.json' % savedir) as f:
    word2idx = json.load(f)
  npz = np.load('%s/weights.npz' % savedir)
  W = npz['arr_0']
  V = npz['arr_1']
  return word2idx, W, V

def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape
    # don't actually use pos2 in calculation, just print what's expected
    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("Sorry")

    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]
    
    vec = p1 -n1 + n2
    
    distances = pairwise_distances(vec.reshape(1,D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]
    # pick one that's not p1, n1, or n2
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1,neg1,neg2)]
    best_idx = list(filter(lambda x: x not in keep_out,idx.tolist()))[0]
    '''
    for i in idx:
        if idx not in keep_out:
            best_idx = i
            break
    '''
    print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[best_idx], neg2))    
    print("closest 10:")
    
    for i in idx:
        print(idx2word[i],distances[i])
    print("dist to %s:" % pos2, cos_dist(p2, vec))

def test_model(word2idx, W, V):
  # there are multiple ways to get the "final" word embedding
  # We = (W + V.T) / 2
  # We = W

  idx2word = {i:w for w, i in word2idx.items()}

  for We in (W, (W + V.T) / 2):
    print("**********")

    analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
    '''
    analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
    analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
    analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
    analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
    analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
    analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
    analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
    analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
    analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
    analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
    analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
    analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
    analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
    analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
    analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
    analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
    analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
    analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
    analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
    analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
    analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
    analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
    analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)
    '''
train_model('/home/rishabh/Rishabh/MachineLearning/NLP-2_code/word2vec_numpy/w2v_model')
word2idx, W, V = load_model('/home/rishabh/Rishabh/MachineLearning/NLP-2_code/word2vec_numpy/w2v_model')    
test_model(word2idx, W, V)       
            
