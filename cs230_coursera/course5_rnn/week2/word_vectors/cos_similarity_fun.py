import numpy as np
import sklearn
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

from w2v_utils import read_glove_vecs


def cosine_similarity_simple(u, v):
    return np.dot(u, v) / (norm(u) * norm(v))


def cosine_similarity_sklearn(u, v):
    return cosine_similarity(u.reshape(1, -1), v.reshape(1, -1)).ravel()[0]


if __name__ == '__main__':
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
