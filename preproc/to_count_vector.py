from sklearn.feature_extraction.text import CountVectorizer
from os.path import join as pjoin
import time
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz

def main():
    start_time = time.time()
    with open(pjoin('..', 'data', 'cut-labeled.txt')) as f:
        matrix = f.readlines()
    print('sentence count: {}'.format(len(matrix)))
    print('Loading takes {}s'.format(time.time() - start_time))

    start_time = time.time()
    sentences = []
    tags = []

    start_time = time.time()
    for line in matrix:
        tag, msg = line.split('\t')
        tags.append(tag)
        sentences.append(msg.strip())
    print('Transforming takes {}s'.format(time.time() - start_time))

    start_time = time.time()
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(sentences)
    print('Vectorizing takes {}s'.format(time.time() - start_time))

    with open(pjoin('..', 'data', 'vec-count-train-features.txt'), 'w') as f:
        for name in vectorizer.get_feature_names():
            f.write('{}\n'.format(name))

    save_npz(pjoin('..', 'data', 'vec-count-train-X.npz'), X)

if __name__ == '__main__':
    main()
