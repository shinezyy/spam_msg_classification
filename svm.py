from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from os.path import join as pjoin
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix


start_time = time.time()
train_data_file = pjoin('data', 'vec-train-small.txt')
df = pd.read_csv(train_data_file, sep=',', header=None)
print('Loading takes {}s'.format(time.time() - start_time))
matrix = df.values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    matrix[:, :-1], matrix[:, -1], test_size=0.2, random_state=0)

start_time = time.time()
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
print('Training takes {}s'.format(time.time() - start_time))
# print(clf.score(X_test, y_test))
y_pred = clf.predict(X_test)

print(y_test, y_pred)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
print("TN: {}, FP: {}".format(TN, FP))
print("FN: {}, TP: {}".format(FN, TP))

