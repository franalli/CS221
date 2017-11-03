import glob
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/train')
y_train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_train')
val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/val')
y_val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_val')

predictions = []

clf = svm.SVC(C=1000.0, cache_size=8000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=2, gamma=1e-6, kernel='rbf',
    max_iter=-1, probability=False, random_state=0, shrinking=True,
    tol=0.001, verbose=False)

clf.fit(train,y_train)

val_predictions = clf.predict(val)
train_predictions = clf.predict(train)

val_acc = np.mean(val_predictions == y_val)
train_acc = np.mean(train_predictions == y_train)

print(confusion_matrix(val_predictions,y_val))
print 'train acc:{}'.format(train_acc)
print 'val acc:{}'.format(val_acc)