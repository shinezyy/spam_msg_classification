import time
from os.path import join as pjoin
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def evaluate(X_train, X_test, y_train, y_test, clf):
    start_time = time.time()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    weights = np.ones_like(y_train)
    weights[y_train == 1] = 3

    clf.fit(X_train, y_train, sample_weight=weights)
    print('Training takes {}s'.format(time.time() - start_time))

    start_time = time.time()
    y_pred = clf.predict(X_test)
    y_pred = np.array([int(x > 0.5) for x in y_pred])
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    print("TN: {}, FP: {}".format(TN, FP))
    print("FN: {}, TP: {}".format(FN, TP))
    print('recall: {}'.format(TP / (FN + TP)))
    print('precision: {}'.format(TP / (FP + TP)))
    print('accuracy: {}'.format((TP + TN) / (FN + FP + TP + TN)))

    print('Prediction takes {}s'.format(time.time() - start_time))


def test_doc2vec(clf):
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
        matrix[:, :-1], matrix[:, -1], test_size=0.2, random_state=0),
        clf)


def test_sparse(x_file, y_file, random_seed: int, clf):
    """
    Wrapper for sparse representation
    :param random_seed: fixed random seed to walk around BUG in xgboost...
    :param x_file: npz file containing instances in sparse matrix
    :param y_file: labels for all instances
    """
    start_time = time.time()
    X = load_npz(x_file)
    y = pd.read_csv(y_file, header=None).values
    print('Loading takes {}s'.format(time.time() - start_time))
    evaluate(*train_test_split(
        X, y, test_size=0.2, random_state=random_seed),
        clf)



def new_GBC():
    clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=1.0,
            max_depth=5,
            random_state=0
            )
    return clf


def new_AdaBoost():
    base_clf = DecisionTreeClassifier(max_depth=5)
    clf = AdaBoostClassifier(
            base_estimator=base_clf,
            n_estimators=200
            )
    return clf


if __name__ == '__main__':
    test_sparse(pjoin('..', 'data', 'hash-vec-train.npz'),
                pjoin('..', 'data', 'labels.txt'), 1, new_AdaBoost())
    test_sparse(pjoin('..', 'data', 'tfidf-vec-train.npz'),
                pjoin('..', 'data', 'labels.txt'), 7, new_AdaBoost())
