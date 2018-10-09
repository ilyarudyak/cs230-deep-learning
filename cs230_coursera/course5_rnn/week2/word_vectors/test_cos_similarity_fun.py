import numpy as np

from w2v_utils import read_glove_vecs
from cos_similarity_fun import cosine_similarity_sklearn


def test_cosine_similarity_sklearn():
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    ball = word_to_vec_map["ball"]
    crocodile = word_to_vec_map["crocodile"]
    france = word_to_vec_map["france"]
    italy = word_to_vec_map["italy"]
    paris = word_to_vec_map["paris"]
    rome = word_to_vec_map["rome"]

    assert np.isclose(cosine_similarity_sklearn(ball, crocodile), 0.274392462614)
    assert np.isclose(cosine_similarity_sklearn(france - paris, rome - italy), -0.675147930817)
