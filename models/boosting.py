import xgboost
from sklearn.model_selection import train_test_split
from os.path import join as pjoin
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from pprint import pprint


start_time = time.time()
train_data_file = pjoin('..', 'data', 'vec-train.txt')
df = pd.read_csv(train_data_file, sep=',', header=None)
print('Loading takes {}s'.format(time.time() - start_time))
matrix = df.values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    matrix[:, :-1], matrix[:, -1], test_size=0.2, random_state=0)

start_time = time.time()
weights = np.ones_like(y_train)
weights[y_train == 1] = 10
train_matrix = xgboost.DMatrix(X_train, label=y_train, weight=weights)
param = {'max_depth': 5, 'eta': 1, 'silent': 1,
         'objective': 'binary:logistic', 'nthread': 1}
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

print('Prediction takes {}s'.format(time.time() - start_time))

