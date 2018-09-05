#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 01:32:41 2018

@author: rishabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from gensim.models import KeyedVectors

train = pd.read_csv('/home/rishabh/Downloads/Datasets-Advance-Nlp/r8-train-all-terms.txt',header=None,sep='\t')
test = pd.read_csv('/home/rishabh/Downloads/Datasets-Advance-Nlp/r8-test-all-terms.txt',header=None,sep='\t')
train.columns = ['label','content']
test.columns = ['label','content']

class GloveVectorizer:
    def __init__(self):
        print('Loading Glove word vectors')
        word2vec = {}
        embedding = []
        idx2word = []
        with open('/home/rishabh/Downloads/Datasets-Advance-Nlp/glove.6B/glove.6B.50d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:],dtype='float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
        print('Found %s word vectors.'%len(word2vec))

        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)} 
        self.V, self.D = self.embedding.shape
        
    def fit(self, data):
        pass
    def transform(self, data):
        X = np.zeros((len(data),self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)                   
                if len(vecs) > 0:
                    vecs = np.array(vecs)
                    X[n] = vecs.mean(axis=0)
                else:
                    emptycount += 1
                n +=1
            print("Number of samples with no words found %s / %s",(emptycount, len(data)))
            return X
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
class Word2VecVectorizer:
    def __init__(self):
        print('Loading in word vectors...')
        self.word_vectors = KeyedVectors.load_word2vec_format('',binary=True)
        print("Finished loading in word Vectors")
        
    def fit(self,data):
        pass 
    
    def transform(self,data):
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]
        
        X = np.zeros((len(data),self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m = m + 1
                except KeyError:
                    pass   
             if len(vecs) > 0:
                vecs = np.array(vecs) 
                X[n] = vecs.mean(axis=0)
             else:
                 emptycount += 1
             n += 1 
             print('Number of samples with no words found:%s / %s'%(emptycount, len(data)))
             return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


vectorizer = GloveVectorizer()

Xtrain = vectorizer.fit_transform(train.content)
Ytest = test.label