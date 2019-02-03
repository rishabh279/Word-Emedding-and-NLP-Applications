from sklearn.metrics.pairwise import pairwise_distances


def find_analogies(w1, w2, w3, We, word2idx, idx2word):

    V, D = We.shape
    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    for dist in ('euclidean','cosine'):
        distances = pairwise_distances(v0.reshape(1,D), We, metric=dist).reshape(V)
        idx = distances.argsort()[:4]
        best_idx = -1
        keep_out = [word2idx[w] for w in (w1, w2, w3)]
        for i in idx:
            if i not in keep_out:
                best_idx = i
                break
        best_word = idx2word[best_idx]

        print('Closest match by', dist, "distance", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)
