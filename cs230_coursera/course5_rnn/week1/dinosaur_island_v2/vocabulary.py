import numpy as np


def get_vocabulary(filename='dinos.txt'):
    with open(filename, 'r') as f:
        dinos_str = f.read()
        vocabulary = sorted(list(set(dinos_str.lower())))

    vocab_to_ix = {ch: i for i, ch in enumerate(vocabulary)}
    ix_to_vocab = {i: ch for i, ch in enumerate(vocabulary)}

    return vocab_to_ix, ix_to_vocab
