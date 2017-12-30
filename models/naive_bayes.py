from sklearn.model_selection import train_test_split
from os.path import join as pjoin
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from scipy.sparse import load_npz

# This script evaluates a bunch of naive bayes classifiers
# Prerequisities:
#   cut_train.py
#   doc2vec.py
#   to_labels.py
#   to_count_vector.py
#   tf_vectorizer.py

def evaluate(name, clf, data_name, X_train, y_train, X_test, y_test):
    if name == 'MultinomialNB' and (data_name == 'doc2vec' or data_name == 'hashing'):
        # Cannot handle negative vector
        return
    if name == 'GaussianNB' and data_name != 'doc2vec':
        return
    if name == 'BernoulliNB' and data_name == 'doc2vec':
        return

    print('{} on {}'.format(name, data_name))
    start_time = time.time()
    if data_name != 'doc2vec':
        y_train = y_train.flatten()

    clf.fit(X_train, y_train)
    print('Training takes {}s'.format(time.time() - start_time))
    start_time = time.time()
    y_pred = clf.predict(X_test)
    print('Prediction takes {}s'.format(time.time() - start_time))
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    print("TN: {}, FP: {}".format(TN, FP))
    print("FN: {}, TP: {}".format(FN, TP))
    print("precision: {}".format(TP * 1.0 / (TP + FP)))
    print("recall: {}".format(TP * 1.0 / (TP + FN)))


start_time = time.time()

train_vec_data_file = pjoin('..', 'data', 'vec-train.txt')
mat_vec = pd.read_csv(train_vec_data_file, sep=',', header=None).values.astype(np.float32)

train_labels_file = pjoin('..', 'data', 'labels.txt')
train_cnt_y = pd.read_csv(train_labels_file, sep=',', header=None).values

train_cnt_data_file = pjoin('..', 'data', 'vec-count-train-X.npz')
train_cnt_X = load_npz(train_cnt_data_file)

train_tfidf_data_file = pjoin('..', 'data', 'tfidf-vec-train.npz')
train_tfidf_X = load_npz(train_tfidf_data_file)

print('Loading takes {}s'.format(time.time() - start_time))

clfs = [
        [GaussianNB(), 'GaussianNB'],
        [BernoulliNB(), 'BernoulliNB'],
        [MultinomialNB(), 'MultinomialNB'],
        ]

train_data = [
        ['doc2vec',    *train_test_split(mat_vec[:, :-1],   mat_vec[:, -1], test_size=0.2, random_state=0)],
        ['tfidf',      *train_test_split(train_tfidf_X,     train_cnt_y,    test_size=0.2, random_state=0)],
        ['occurrence', *train_test_split(train_cnt_X,       train_cnt_y,    test_size=0.2, random_state=0)],
        ]



for clf, name in clfs:
    for data_name, X_train, X_test, y_train, y_test in train_data:
        evaluate(name, clf, data_name, X_train, y_train, X_test, y_test)
