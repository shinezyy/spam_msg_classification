import time
from os.path import join as pjoin

import pandas as pd
import numpy as np

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.sparse import load_npz


def evaluate(X_train, X_test, y_train, y_test):
    start_time = time.time()
    weights = np.ones_like(y_train)
    # weights[y_train == 1] = 10
    train_matrix = xgboost.DMatrix(X_train, label=y_train, weight=weights)
    param = {'max_depth': 5, 'eta': 1, 'silent': 1,
             'objective': 'binary:logistic', 'nthread': 2}
    clf = xgboost.train(params=param, dtrain=train_matrix, num_boost_round=30)
    print('Training takes {}s'.format(time.time() - start_time))

    start_time = time.time()
    test_matrix = xgboost.DMatrix(X_test, label=y_test)
    y_pred = clf.predict(test_matrix)
    y_pred = np.array([int(x > 0.5) for x in y_pred])
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    print("TN: {}, FP: {}".format(TN, FP))
    print("FN: {}, TP: {}".format(FN, TP))
    print('recall: {}'.format(TP / (FN + TP)))
    print('precision: {}'.format(TP / (FP + TP)))
    print('accuracy: {}'.format((TP + TN) / (FN + FP + TP + TN)))

    print('Prediction takes {}s'.format(time.time() - start_time))


def test_doc2vec():
    """
    Wrapper for dense representation
    Only doc2vec uses dense representation
    """
    start_time = time.time()
    train_data_file = pjoin('..', 'data', 'vec-train.txt')
    df = pd.read_csv(train_data_file, sep=',', header=None)
    print('Loading takes {}s'.format(time.time() - start_time))
    matrix = df.values.astype(np.float32)
    evaluate(*train_test_split(
        matrix[:, :-1], matrix[:, -1], test_size=0.2, random_state=0))


def test_sparse(x_file, y_file):
    """
    Wrapper for sparse representation
    :param x_file: npz file containing instances in sparse matrix
    :param y_file: labels for all instances
    """
    start_time = time.time()
    X = load_npz(x_file)
    y = pd.read_csv(y_file, header=None).values
    print('Loading takes {}s'.format(time.time() - start_time))
    evaluate(*train_test_split(
        X, y, test_size=0.2))


if __name__ == '__main__':
    test_sparse(pjoin('..', 'data', 'tfidf-vec-train.npz'),
                pjoin('..', 'data', 'labels.txt'))
    test_sparse(pjoin('..', 'data', 'hash-vec-train.npz'),
                pjoin('..', 'data', 'labels.txt'))
