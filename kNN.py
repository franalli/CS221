import glob
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed

train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/train')
y_train = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_train')
val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/val')
y_val = np.loadtxt('/home/franalli/Documents/UrbanSound8K/y_val')

# train -= np.mean(train,axis=0)
# val -= np.mean(val,axis=0)

K = [i for i in range(1,100)]
for k in K:
    predictions = []
    for i in range(len(val)):
        distances = np.sum(np.abs(train-val[i]),axis=1)
        neighbor_indeces = np.argsort(distances)[:k]
        neighbor_labels = y_train[neighbor_indeces]
        neighbor_count = collections.defaultdict(int)
        for label in neighbor_labels:
            neighbor_count[label] +=1
        max_label = list(neighbor_count)[0]
        for label in neighbor_count:
            if neighbor_count[label] > neighbor_count[max_label]:
                max_label = label
        predictions.append(max_label)

    acc = np.mean(predictions == y_val)
    print 'k:{},acc:{}'.format(k,acc)