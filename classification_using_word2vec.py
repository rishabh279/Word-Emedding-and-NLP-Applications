#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 01:32:41 2018

@author: rishabh
"""

import numpy as np
import pandas as pd
import matlpotlib.pyplot as plt 

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
        idxword = []
        with open('') as f:
            for line in f:
                values = line.split()
                word = values[0]