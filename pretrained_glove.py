#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 22:09:07 2018

@author: rishabh
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def dist1(a,b):
    return np.linalg.norm(a-b)

def dist2(a,b):
    return 1-a.dot(b)/(np.linalg.norm(a)*np.linalg.norm.norm(b))

dist,metric = dist2, 'cosine'

def find_analogies(w1,w2,w3):
    for w in (w1,w2,w3):
        if w not in word2vec:
            print('Not in dcitionary')
            return 
     
    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king - man + woman
    
    distances = pairwise_distances(v0.reshape(1,D),embedding,metric=metric).reshape(V)
    idxs = distances.argsort()[:4]
    for idx in idxs:
        if idx2word[idx] not in (w1,w2,w3):
            best_word = idx2word[idx]
    print(best_word)
    
    
def nearest_neighbors(w,n=5):
    if w not in word2vec:
        print('Not in dictionary')
        return 
    
    v = word2vec[w]
    distances = pairwise_distances(v.reshape(1,D),embedding,metric=metric).reshape(V)
    idxs = distances.argsort()[1:n+1]
    for idx in idxs:
        print(idx2word[idx])

print('Loading word vectors....')
word2vec = {}
embedding = [] 
idx2word = []
with open('/home/rishabh/Downloads/Glove/glove.6B/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
embedding = np.array(embedding)
V , D = embedding.shape

w1,w2,w3='king','man','woman'    
for w in (w1,w2,w3):
    
    if w not in word2vec:
        print('Not in dcitionary')
        return 
     
king = word2vec[w1]
man = word2vec[w2]
woman = word2vec[w3]
v0 = king - man + woman

v0=v0.reshape(1,D)
distances = pairwise_distances(v0.reshape(1,D),embedding,metric=metric).reshape(V)
idxs = distances.argsort()[:4]
for idx in idxs:
    if idx2word[idx] not in (w1,w2,w3):
        best_word = idx2word[idx]
print(best_word)    

              
find_analogies('king', 'man', 'woman')
find_analogies('france', 'paris', 'london')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')
find_analogies('france', 'french', 'english')
find_analogies('japan', 'japanese', 'chinese')
find_analogies('japan', 'japanese', 'italian')
find_analogies('japan', 'japanese', 'australian')
find_analogies('december', 'november', 'june')
find_analogies('miami', 'florida', 'texas')
find_analogies('einstein', 'scientist', 'painter')
find_analogies('china', 'rice', 'bread')
find_analogies('man', 'woman', 'she')
find_analogies('man', 'woman', 'aunt')
find_analogies('man', 'woman', 'sister')
find_analogies('man', 'woman', 'wife')
find_analogies('man', 'woman', 'actress')
find_analogies('man', 'woman', 'mother')
find_analogies('heir', 'heiress', 'princess')
find_analogies('nephew', 'niece', 'aunt')
find_analogies('france', 'paris', 'tokyo')
find_analogies('france', 'paris', 'beijing')
find_analogies('february', 'january', 'november')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')

nearest_neighbors('king')
nearest_neighbors('france')
nearest_neighbors('japan')
nearest_neighbors('einstein')
nearest_neighbors('woman')
nearest_neighbors('nephew')
nearest_neighbors('february')
nearest_neighbors('rome')