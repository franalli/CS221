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

train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/train')
y_train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_train')
val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/val')
y_val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_val')

predictions = []
clf = svm.SVC(C=0.1, cache_size=8000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=2, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=0, shrinking=True,
    tol=0.001, verbose=False)

# clf= svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=0.01, 
# 	multi_class='ovr', fit_intercept=True, intercept_scaling=1, 
# 	class_weight=None, verbose=0, random_state=0, max_iter=100)

clf.fit(train,y_train)

val_predictions = clf.predict(val)
train_predictions = clf.predict(train)

val_acc = np.mean(val_predictions == y_val)
train_acc = np.mean(train_predictions == y_train)

print 'train acc:{}'.format(train_acc)
print 'val acc:{}'.format(val_acc)