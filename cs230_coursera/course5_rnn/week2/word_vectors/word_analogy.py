import numpy as np

from cos_similarity_fun import cosine_similarity_sklearn


def get_similarity(word, words, word_to_vec_map):
    word_vectors = [word_to_vec_map[word] for word in words]
    return cosine_similarity_sklearn(word_vectors[1] - word_vectors[0],
                                     word_to_vec_map[word] - word_vectors[2])


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c,
    as measured by cosine similarity
    """
    words = [word_a, word_b, word_c]
    words = [word.lower() for word in words]

    similarities_dict = {word: get_similarity(word, words, word_to_vec_map)
                         for word in word_to_vec_map}
    return max(similarities_dict, key=similarities_dict.get)






