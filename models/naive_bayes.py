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
#   1. cut_train.py
#   2. doc2vec.py
#   1. to_labels.py
#   2. to_count_vector.py

def evaluate(name, clf, X_train, y_train, X_test, y_test):
    print(name)
    start_time = time.time()
    if name == 'MultinomialNB':
        y_train = y_train.flatten()
    clf.fit(X_train, y_train)
    print('Training takes {}s'.format(time.time() - start_time))
    print(clf.score(X_test, y_test))
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

train_cnt_data_file = pjoin('..', 'data', 'vec-count-train-X.npz')
train_labels_file = pjoin('..', 'data', 'labels.txt')
train_cnt_X = load_npz(train_cnt_data_file)
train_cnt_y = pd.read_csv(train_labels_file, sep=',', header=None).values
print('Loading takes {}s'.format(time.time() - start_time))

clfs = [
        [GaussianNB(), 'GaussianNB',
            *train_test_split(mat_vec[:, :-1], mat_vec[:, -1], test_size=0.2, random_state=0)],
        [BernoulliNB(), 'BernoulliNB',
            *train_test_split(mat_vec[:, :-1], mat_vec[:, -1], test_size=0.2, random_state=0)],
        [MultinomialNB(), 'MultinomialNB',
            *train_test_split(train_cnt_X, train_cnt_y, test_size=0.2, random_state=0)],
        ]


for clf, name, X_train, X_test, y_train, y_test in clfs:
    evaluate(name, clf, X_train, y_train, X_test, y_test)
