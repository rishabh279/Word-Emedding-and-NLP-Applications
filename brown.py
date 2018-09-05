#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:17:49 2018

@author: rishabh
"""
from nltk.corpus import brown
import operator

KEEP_WORDS = set([
  'king', 'man', 'queen', 'woman',
  'italy', 'rome', 'france', 'paris',
  'london', 'britain', 'england',
])

def get_sentences():
    return brown.sents()

def get_sentences_with_word2idx():
    sentences = get_sentences()
    indexed_sentences = []
    i=2
    word2idx = {'START':0,'END':1}
    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            if token not in word2idx:
                word2idx[token] = i
                i = i+1
            indexed_sentence.append(word2idx[token])
    indexed_sentences.append(indexed_sentence)
        
def get_sentences_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
    sentences = get_sentences()
    indexed_sentences = []
    
    i = 2
    word2idx = {'START':0, 'END':1}
    idx2word = ['START', 'END'] 
    
    word_idx_count = {
            0: float('inf'),
            1: float('inf'),
            } 
    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            token = token.lower()
            if token not in word2idx:
                idx2word.append(token)
                word2idx[token] = i
                i = i + 1
            idx = word2idx[token] 
            word_idx_count[idx] = word_idx_count.get(idx,0)+1
            
            indexed_sentence.append(idx)
            
        indexed_sentences.append(indexed_sentence)
    
    for word in keep_words:
        word_idx_count[word2idx[word]] = float('inf')
        
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True) 
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        print(idx)
        word = idx2word[idx]
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx = new_idx+1
    
    word2idx_small['UNKNOWN'] = new_idx
    unknown = new_idx

    sentences_small = []
    for sentence in indexed_sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)
    return sentences_small, word2idx_small
        
