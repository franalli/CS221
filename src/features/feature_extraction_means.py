#copyright: https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20NN.ipynb

import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    mel = np.array(librosa.feature.melspectrogram(X, sr=sample_rate))
    D,T = np.shape(mel)
    feature = np.empty((D))
    for t in range(T):
        feature = np.hstack([feature,mel[:,t]])


    return feature[D:]

def parse_audio_files(parent_dir,sub_dirs,num_ex,file_ext="*.wav"):
    labels = np.empty(0)
    features = np.empty((0,17024))
    for label, sub_dir in enumerate(sub_dirs):
        i = 0
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            i +=1
            try:
              mel= extract_feature(fn)
              print np.shape(mel)
            except Exception as e:
              print "Error encountered while parsing file: ", fn
              continue

    return np.array(features), np.array(labels, dtype = np.int)

parent_dir = '/home/franalli/Documents/UrbanSound8K/audio/'
train_sub_dirs = ["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8"]
val_sub_dirs = ["fold9"]
test_sub_dirs = ["fold10"]

train_features, train_labels = parse_audio_files(parent_dir,train_sub_dirs,1)
# val_features, val_labels = parse_audio_files(parent_dir,val_sub_dirs,10000)
# test_features, test_labels = parse_audio_files(parent_dir,test_sub_dirs,100)

np.savetxt('/home/franalli/Documents/UrbanSound8K/train',train_features)
np.savetxt('/home/franalli/Documents/UrbanSound8K/val',val_features)
# np.savetxt('/home/franalli/Documents/UrbanSound8K/test',test_features)
np.savetxt('/home/franalli/Documents/UrbanSound8K/y_train',train_labels)
np.savetxt('/home/franalli/Documents/UrbanSound8K/y_val',val_labels)
# np.savetxt('/home/franalli/Documents/UrbanSound8K/y_test',test_labels)