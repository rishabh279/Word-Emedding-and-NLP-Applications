# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances

import os
import string
import sys

def get_text8():
    path = '/home/rishabh/Rishabh/MachineLearning/NLP-2_code/text8'
    words = open(path).read()
    word2idx = {}
    sents = [[]]
    count = 0
    for word in words.split():
        if word not in word2idx:
            word2idx[word] = count
            count += 1
        sents[0].append(word2idx[word])
    return sents,word2idx    

def train_model(savedir):
    #get the data
    sentences, word2idx = get_text8()
    
    # number of unique words
    vocab_size = len(word2idx)
    
    #config
    window_size = len(word2idx)
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negatives = 5
    samples_per_epoch = int(1e5)
    epochs = 20
    D = 50
    
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs
    
    p_neg = get_negative_sampling_distribution(sentences)
    
    #params
    W = np.random.randn(vocab_size, D).astype(np.float32)# input-to-hidden
    V = np.random.randn(D, vocab_size).astype(np.float32)# hidden-to-output
    
    # create the model
    tf_input = tf.placeholder(tf.int32,shape=(None,))
    tf_negword = tf.placeholder(tf.int32, shape=(None,))
    tf_context = tf.placeholder(tf.int32, shape=(None,))
    tfW = tf.Variable(W)
    tfV = tf.Variable(V.T)
    
    def dot(A,B):
        C = A * B
        return tf.reduce_sum(C, axis=1)
    
    emb_input  = tf.nn.embedding_lookup(tfW, tf_input)# 1 * D
    emb_output = tf.nn.embedding_lookup(tfV, tf_context)# N * D
    
    # correct middle word output
    correct_output = dot(emb_input, emb_output)
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones(tf.shape(correct_output)), logits=correct_output)
    
    #incorrect middle word output 
    emb_input = tf.nn.embedding_lookup(tfW, tf_negword)
    incorrect_output = dot(emb_input,emb_output)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros(tf.shape(incorrect_output)), logits=incorrect_output)
    
    #total loss
    loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)
    
    #optimizer
    train_op = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
    
    session = tf.Session()
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    
    costs = []
    
    # for subsampling each sentence
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)
    
    #train the model
    for epoch in range(epochs):
        # randomly order sentences so we don't always see
        # sentences in the same order
        np.random.shuffle(sentences)
        # accumulate the cost
        cost = 0
        counter = 0
        inputs = []
        targets = []
        negwords = []
        for sentence in sentences:
            sentence = [w for w in sentence if np.random.random() < (1 - p_drop[w])]
        if len(sentence) < 2:
            continue
        
        randomly_ordered_positions = np.random.choice(
                len(sentence),
                size=len(sentence),
                replace=False,)
        
        for j, pos in enumerate(randomly_ordered_positions):
            word = sentence[pos]
            
            context_words = get_context(pos, sentence, window_size)
            neg_word = np.random.choice(vocab_size, p=p_neg)
            
            n = len(context_words)
            inputs += [word]*n
            negwords += [neg_word]*n
            targets += context_words
            
            if len(inputs) >= 128:
                _,c = session.run(
                    (train_op, loss),
                    feed_dict={
                        tf_input: inputs,
                        tf_negword: negwords,
                        tf_context: targets,
                    }
                )
                cost += c
                
                #reset
                inputs = []
                targets = []
                negwords = []
            counter += 1
            if counter % 100 == 0:
               sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
               sys.stdout.flush()
        costs.append(cost)
        
        learning_rate -= learning_rate_delta
    plt.plot(cost)
    
    # get the params
    W, VT = session.run((tfW, tfV))
    V = VT.T
    
    # save the model
    if not os.path.exists(savedir):
        os.mkdir(savedir)    
    
    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)        
    
    np.savez('%s/weights.npz' % savedir, W, V) 
    
    return word2idx, W, V
                
            
def get_context(pos, sentence, window_size):
    # input:
    # a sentence of the form: x x x x c c c pos c c c x x x x
    # output:
    # the context word indices: c c c c c c
    start = max(0,pos - window_size)
    end = min(len(sentence), pos + window_size)    
        
    context = []
    for ctx_pos, ctx_word_idx in enumerate(sentence[start:end],start=start):
        if ctx_pos != pos:
            context.append(ctx_word_idx)
    
    return context
        
def get_negative_sampling_distribution(sentences):
    # Pn(w) = prob of word occuring
    # we would like to sample the negative samples
    # such that words that occur more often
    # should be sampled more often
    word_count = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
    # vocab_size
    V = len(word_count)
    p_neg = np.zeros(V)
    for j in range(V):
        p_neg[j] = word_count[j] ** 0.75
        
    p_neg = p_neg / p_neg.sum()
    
    assert(np.all(p_neg > 0))
    return p_neg


def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape
    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("Sorry")
            
    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]

    vec = p1 - n1 +n2
    
    distances = pairwise_distances(vec.reshape(1,D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    for i in idx:
        if i not in keep_out:
          best_idx = i
          break
    print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[idx[0]], neg2))
    print('Closest 10:')
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
    analogy('japan', 'sushi', 'england', 'bread', word2idx, idx2word, We)
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
 
word2idx, W, V = train_model('/home/rishabh/Rishabh/MachineLearning/NLP-2_code/word2vec_tensorflow/w2v_tf')
test_model(word2idx,W,V)    
    