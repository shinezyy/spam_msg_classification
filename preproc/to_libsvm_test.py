from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from os.path import join as pjoin
import pandas as pd
import numpy as np
import time
from sklearn.datasets import dump_svmlight_file


def main():
    """
    Convert vec-test.txt to sparse format for libsvm
    This file is usually useless for most of members
    :return: None
    """
    start_time = time.time()

    train_data_file = pjoin('..', 'data', 'vec-test.txt')
    df = pd.read_csv(train_data_file, sep=',', header=None)
    matrix = df.values.astype(np.float32)

    print('Loading takes {}s'.format(time.time() - start_time))

    X = matrix[:, :-1]
    y = matrix[:, -1]

    start_time = time.time()

    out_data_file = pjoin('..', 'data', 'test_data_libsvm_full.dat')
    dump_svmlight_file(X, y, out_data_file)

    print('Converting takes {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
