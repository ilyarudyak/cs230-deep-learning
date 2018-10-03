import numpy as np
from utils import *
import random


def get_data():
    data = open('dinos.txt', 'r').read()
    data = data.lower()
    chars = list(set(data))
    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    return char_to_ix, ix_to_char
