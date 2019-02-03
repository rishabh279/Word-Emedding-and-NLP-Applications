import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
# from util import init_weight
from sklearn.metrics import f1_score
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
import sys


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


class Data:

    def __init__(self):
        self.word2idx = {}
        self.tag2idx = {}
        self.word_idx = 1
        self.tag_idx = 1
        self.xtrain = []
        self.ytrain = []
        self.currentx = []
        self.currenty = []

    def get_data_training(self, filename, split_sequences=False):
        for line in open('{}'.format(filename)):
            line = line.strip()
            if line:
                r = line.split()
                word, tag, _ = r
                if word not in self.word2idx:
                    self.word2idx[word] = self.word_idx
                    self.word_idx += 1
                    self.currentx.append(word2idx[word])

                if tag not in self.tag2idx:
                    self.tag2idx[tag] = self.tag_idx
                    self.tag_idx += 1
                    self.currenty.append(self.tag2idx[tag])
            elif split_sequences:
                self.xtrain.append(self.currentx)
                self.ytrain.append(self.currenty)
                self.currentx = []
                self.currenty = []

        if not split_sequences:
            self.xtrain = self.currentx
            self.ytrain = self.currenty

        return self.xtrain, self.ytrain, self.word2idx

    def get_data_test_pos(self):
        xtest = []
        ytest = []
        currentx = []
        currenty = []
        for line in open('data/chunking/test.txt'):
            line = line.strip()
            if line:
                r = line.split()
                word, tag, _ = r
                if word in self.word2idx:
                    currentx.append(self.word2idx[word])
                else:
                    currentx.append(self.word_idx)

        return xtest, ytest

    def get_data_test_ner(self):
        xtrain, ytrain, _ = self.get_data_training()
        ntest = int(0.3*len(xtrain))
        xtest = xtrain[:ntest]
        ytest = ytrain[:ntest]
        xtrain = xtrain[ntest:]
        ytrain = ytrain[ntest:]
        return xtrain, ytrain, xtest, ytest


def flatten(l):
    return [item for sublist in l for item in sublist]


d = Data()
xtrain, ytrain, word2idx = d.get_data('data/chunking/train.text', split_sequences=True)
xtest, ytest = d.get_data_test_pos()
# for ner
# xtrain, ytrain, xtest, ytest = d.get_data_test_ner()
v = len(word2idx) + 2
k = len(set(flatten(ytrain)) | set(flatten(ytest))) + 1


epochs = 20
learning_rate = 1e-2
mu = 0.99
batch_sz =32
hidden_layer_size = 10
embedding_dim = 10
sequence_length = max(len(x) for x in xtrain + xtest)


xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=sequence_length)
ytrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=sequence_length)
xtest = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=sequence_length)
ytest = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=sequence_length)
print('xtrain shape', xtrain.shape)
print('ytrain shape', ytrain.shape)

inputs = tf.placeholder(tf.int32, shape=(None, sequence_length))
targets = tf.placeholder(tf.int32, shape=(None, sequence_length))
num_samples = tf.shape(inputs)[0]

we = np.random.randn(v, embedding_dim).astype(np.float32)

wo = init_weight(hidden_layer_size, k).astype(np.float32)
bo = np.zeros(k).astype(np.float32)

tfwe = tf.Variable(we)
tfwo = tf.Variable(wo)
tfbo = tf.Variable(bo)

rnn_unit = GRUCell(num_units=hidden_layer_size, activation=tf.nn.relu)

x = tf.nn.embedding_lookup(tfwe, inputs)

x = tf.unstack(x, sequence_length, 1)

outputs, states = get_rnn_output(rnn_unit, x, dtype=tf.float32)

outputs = tf.transpose(outputs, (1, 0, 2))
outputs = tf.reshape(outputs, (sequence_length*num_samples, hidden_layer_size))

logits = tf.matmul(outputs, tfwo) + tfbo
predictions = tf.argmax(logits, 1)
predict_op = tf.reshape(predictions, (num_samples, sequence_length))
labels_flat = tf.reshape(targets, [-1])

cost_op = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels_flat
    )
)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

sess = tf.InterativeSession()
init = tf.global_variables_initializer()
sess.run(init)

costs = []
n_batches = len(ytrain) // batch_sz
for i in range(epochs):
    n_total = 0
    n_correct = 0

    xtrain, ytrain = shuffle(xtrain, ytrain)
    cost = 0

    for j in range(n_batches):
        x = xtrain[j*batch_sz:(j+1)*batch_sz]
        y = ytrain[j*batch_sz:(j+1)*batch_sz]

        c, p, _ = sess.run(
            (cost_op, predict_op, train_op),
            feed_dict={inputs: x, targets: y}
        )
        cost += c

        for yi, pi in zip(y, p):
            yii = yi[yi > 0]
            pii = pi[yi > 0]
            n_correct += np.sum(yii == pii)
            n_total += len(yii)

        if j%10 == 0:
            sys.stdout.write(
                "j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
                (j, n_batches, float(n_correct) / n_total, cost)
            )
            sys.stdout.flush()

    p = sess.run(predict_op, feed_dict={inputs: xtest, targets: ytest})
    n_test_correct = 0
    n_test_total = 0
    for yi, pi in zip(ytest, p):
        yii = yi[yi > 0]
        pii = pi[yi > 0]
        n_test_correct += np.sum(yii == pii)
        n_test_total += len(yii)
    test_acc = float(n_test_correct) / n_test_total

    print(
        "i:", i, "cost:", "%.4f" % cost,
        "train acc:", "%.4f" % (float(n_correct) / n_total),
        "test acc:", "%.4f" % test_acc
    )
    costs.append(cost)

