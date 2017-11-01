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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/train')
y_train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_train')
val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/val')
y_val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_val')

predictions = []



clf = RandomForestClassifier(n_estimators=2000, criterion='gini', max_depth=None, 
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, 
    random_state=0, verbose=0, warm_start=False, class_weight=None)

clf.fit(train,y_train)

val_predictions = clf.predict(val)
feature_importance = clf.feature_importances_
train_predictions = clf.predict(train)

val_acc = np.mean(val_predictions == y_val)
train_acc = np.mean(train_predictions == y_train)


print(confusion_matrix(val_predictions,y_val))
print 'train acc:{}'.format(train_acc)
print 'val acc:{}'.format(val_acc)