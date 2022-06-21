import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import os
import itertools

import config


def splitter(seq):
    seqs = []
    for s in seq:
        seqs.append(s)
    return seqs

def preprocessing(df):
    df.loc[:, 'PepSeq'] = df.PepSeq.map(splitter)
    text = list(df.PepSeq)
    
    corpus = list(itertools.chain.from_iterable(text))
    corpus.append('\n')

    print('corpus length:', len(text))

    chars = sorted(list(set(corpus)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


    # cut the text in semi-redundant sequences of config.MAX_LEN characters
    sentences = []
    next_chars = []
    for i, t in enumerate(text):
        if len(t) < config.MAX_LEN:
            sentences.append(t)
            t = t[1:] + ['\n']
            next_chars.append(t)
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), config.MAX_LEN, len(chars)))
    y = np.zeros((len(sentences), config.MAX_LEN, len(chars)))
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1

    for i, next_char in enumerate(next_chars):
        for t, char in enumerate(next_char):
            y[i, t, char_indices[char]] = 1

    return x, y, char_indices, indices_char
