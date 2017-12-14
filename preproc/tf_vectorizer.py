# -*- coding: utf-8 -*-

import numpy as np
from os.path import join as pjoin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from scipy.sparse import save_npz


n_features = 100000
chinese_punctuations = [
    ' ', '，', '。', '？', '；', '"', '～', '【',
    '】', '/', '：', '『', '﹃', '﹄', '』', '《', '》',
    '──', '‘ ', '’', '“ ', '”', ' '
]


def vectorize(inf: str, outf: str,
              vectorizer: HashingVectorizer or TfidfVectorizer or None = None,
              trained_vectorizer: HashingVectorizer or TfidfVectorizer or None = None):
    with open(inf, encoding='utf-8') as f:
        lines = f.read().split('\n')[:-1]
    terms = []
    for line in lines:
        line = line.split('\t')[-1]
        terms.append(line)

    if trained_vectorizer is None:
        vectors = vectorizer.fit_transform(terms)
        save_npz(outf, vectors)
        return
    else:
        vectors = trained_vectorizer.transform(terms)
        save_npz(outf, vectors)


def vectorize_with(vectorizer: HashingVectorizer or TfidfVectorizer, name: str):
    vectorize(pjoin('..', 'data', 'cut-labeled.txt'),
              pjoin('..', 'data', '{}-vec-train.npz'.format(name)),
              vectorizer=vectorizer)
    vectorize(pjoin('..', 'data', 'cut-no-label.txt'),
              pjoin('..', 'data', '{}-vec-test.npz'.format(name)),
              trained_vectorizer=vectorizer)


def new_hash_vectorizer():
    return HashingVectorizer(
        n_features=n_features,
        stop_words=chinese_punctuations,
    )


def new_tfidf_vectorizer():
    return TfidfVectorizer(
        max_df=0.8, min_df=2,
        max_features=n_features,
        stop_words=chinese_punctuations
    )


if __name__ == '__main__':
    vectorize_with(new_hash_vectorizer(), 'hash')
    vectorize_with(new_tfidf_vectorizer(), 'tfidf')
