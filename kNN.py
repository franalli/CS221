import glob
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/train')
y_train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_train')
val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/val')
y_val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_val')



for k in range(1,100):
    clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', 
        leaf_size=30, p=1, metric='minkowski', metric_params=None, n_jobs=-1)

    clf.fit(train,y_train)

    val_predictions = clf.predict(val)
    train_predictions = clf.predict(train)

    val_acc = np.mean(val_predictions == y_val)
    train_acc = np.mean(train_predictions == y_train)

    # print(confusion_matrix(val_predictions,y_val))
    print k
    print 'train acc:{}'.format(train_acc)
    print 'val acc:{}'.format(val_acc)
    print