# -*- coding: utf-8 -*-

from os.path import join as pjoin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def vectorize(inf: str, outf: str, model=TfidfVectorizer) -> TfidfVectorizer:
    with open(inf) as f:
        matrix = f.read().split('\n')[:-1]
    n_features = 100
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    vectors = tfidf_vectorizer.fit_transform(matrix)
    print(vectors)
    return tfidf_vectorizer


if __name__ == '__main__':
    vectorize(pjoin('..', 'data', 'cut-labeled.txt'),
              pjoin('..', 'data', 'tfidf-vec-train.txt'))
