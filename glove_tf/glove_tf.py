import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import find_analogies

from brown import get_sentences_with_word2idx_limit_vocab as get_brown

class Glove:
    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10):

        # build co-occurrence matrix
        # paper calls it X, so we will call it X, instead of calling
        # the training data X
        V = self.V
        D = self.D

        if not os.path.exits(cc_matrix):
            X = np.zeros((V,V))
            N = len(sentences)
            print('No of sentences to process:', N)
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print("processed", it, "/", N)
                n = len(sentence)
                for i in range(n):
                    # i just points to which element of the sequence (sentence) we're looking at
                    wi = sentence[wi]

                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_size)

                    # we can either choose only one side as context, or both
                    # here we are doing both

                    # make sure "start" and "end" tokens are part of some context
                    # otherwise their f(X) will be 0 (denominator in bias update)

                    if i - self.context_sz < 0:
                        points = 1.0 / (i + 1)
                        X[wi,0] += points
                        X[0,wi] += points

                    if i + self.context_sz > n:
                        points = 1.0 / (n-i)
                        X[wi,1] += points
                        X[1,wi] += points

                    # left side
                    for j in range(start,i):
                        wj = sentence[j]
                        points = 1 / (i-j)
                        X[wi,wj] += points
                        X[wj,wi] += points

                    #right side
                    for j in range(i+1, end):
                        wj = sentence[j]
                        points = 1 / (j - i)
                        X[wi,wj] += points
                        X[wj,wi] += points
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        print("Max in X:", X.max())

        # weighting
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X >= xmax] = 1

        print("max in f(X):", fX.max())

        # target
        logX = np.log(X + 1)

        print("max in log(X):", logX.max())

        # initialize weights
        W = np.random.randn((V, D)) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        # initialize weights, inputs, targets placeholders
        tfW = tf.Variable(W.astype(np.float32))
        tfb = tf.Variable(b.reshape(V, 1).astype(np.float32))
        tfU = tf.Variable(U.astype(np.float32))
        tfc = tf.Variable(c.reshape(1, V).astype(np.float32))
        tfLogX = tf.placeholder(tf.float32, shape=(V, V))
        tffX = tf.placeholder(tf.float32, shape=(V,V))

        delta = tf.matmul(tfW, tf.transpose(tfU) + tfb + tfc + mu - tfLogX)
        cost = tf.reduce_sum(tffX * delta * delta)
        regularized_cost = cost
        for param in (tfW, tfU):
            regularized_cost += reg * tf.reduce_sum(param * param)

        train_op = tf.train.MomentumOptimizer(
            learning_rate,
            momentum = 0.9,
        ).minimize(regularized_cost)

        init = tf.globa_variables_initalizer()
        session = tf.InteractiveSession()
        session.run(init)

        costs = []
        sentences_indexes = range(len(sentences))
        for epoch in range(epochs):
            c, _ = session.run((cost, train_op), feed_dict={tfLogX: logX, tffX: fX})
            print("epoch:", epoch, "cost:", c)
            costs.append(c)

        self.W, self.U = session.run([tfW, tfU])

        plt.plot(costs)
        plt.show()

    def save(self, fn):
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)

def main(we_file, w2_ifile):
    cc_matrix = "cc_matrix_brown.npy"

    if os.path.exists(cc_matrix):
        with open(w2_ifile) as f:
            word2idx = json.load(f)
        sentences = []
    else:
        keep_words = set([
            'king', 'man', 'woman',
            'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
            'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
            'australia', 'australian', 'december', 'november', 'june',
            'january', 'february', 'march', 'april', 'may', 'july', 'august',
            'september', 'october',
        ])
        sentences, word2idx = get_brown(n_vocab=5000, keep_words=keep_words)
        with open(w2_ifile, 'w') as f:
            json.dump(word2idx, f)

    V = len(word2idx)
    model = Glove(200, V, 10)

    model.fit(sentences, cc_matrix=cc_matrix, epochs=20)
    model.save(we_file)


if __name__ == '__main__':
    we = 'glove_model_50.npz'
    w2i = 'glove_word2idx_50.json'

    main(we, w2i)
    npz = np.load(we)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i: w for w, i in word2idx.items()}

    for concat in (True, False):
        print("** concat:", concat)

        if concat:
            We = np.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2

        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)
