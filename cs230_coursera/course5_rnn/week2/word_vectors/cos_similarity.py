import numpy as np
from numpy.linalg import norm

from w2v_utils import read_glove_vecs


def cosine_similarity(u, v):
    return np.dot(u, v) / (norm(u) * norm(v))


if __name__ == '__main__':
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
