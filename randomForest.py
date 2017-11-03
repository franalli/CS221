import glob
import os
import itertools
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



clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, 
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, 
    random_state=0, verbose=0, warm_start=False, class_weight=None)

clf.fit(train,y_train)

val_predictions = clf.predict(val)

# mfccs:40, chroma:12, mel:128, contrast:7, tonnetz:6, zero_crossing_rate:1
# rmse:1, bw:1
feature_importances = clf.feature_importances_
mfccs = np.sum(feature_importances[:40])
chroma = np.sum(feature_importances[40:52])
mel= np.sum(feature_importances[52:180])
contrast= np.sum(feature_importances[180:187])
tonnetz= np.sum(feature_importances[187:193])
zcr = np.sum(feature_importances[193:194])
rmse = np.sum(feature_importances[194:195])
bw = np.sum(feature_importances[195:196])
normalizer = np.max([mfccs,chroma,mel,contrast,tonnetz,zcr,rmse,bw])
objects = ('mfccs', 'chroma','mel','contrast','tonnetz','zcr','rmse','bw')
y_pos = np.arange(len(objects))
performance = [mfccs/normalizer,chroma/normalizer,mel/normalizer,contrast/normalizer,tonnetz/normalizer,zcr/normalizer,rmse/normalizer,bw/normalizer]
plt.bar(y_pos, performance, align='center', alpha=1.0)
plt.xticks(y_pos, objects,fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Scaled Feature Importance',size=25)
plt.title('Relative Importance of Features in Random Forest',size=25)
 
# plt.savefig('RF_feature_importance',dpi='figure')
# plt.show()
train_predictions = clf.predict(train)

val_acc = np.mean(val_predictions == y_val)
train_acc = np.mean(train_predictions == y_train)


CM = confusion_matrix(val_predictions,y_val)
np.savetxt('RF_CM',CM)


print 'train acc:{}'.format(train_acc)
print 'val acc:{}'.format(val_acc)