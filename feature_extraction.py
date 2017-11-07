#copyright: https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20NN.ipynb

import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed

# def extract_feature(file_name):
#     X, sample_rate = librosa.load(file_name)
#     mfccs = np.mean(librosa.feature.mfcc(y=X, S = None,sr=sample_rate, n_mfcc=40).T,axis=0)
#     chroma = np.mean(librosa.feature.chroma_stft(y=X, S=None, sr=sample_rate).T,axis=0)
#     mel = np.mean(librosa.feature.melspectrogram(y=X, S=None,sr=sample_rate).T,axis=0)
#     contrast = np.mean(librosa.feature.spectral_contrast(y=X,S=None, sr=sample_rate).T,axis=0)
#     tonnetz = np.mean(librosa.feature.tonnetz(y=X, sr=sample_rate).T,axis=0)
#     zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0)
#     rmse = np.mean(librosa.feature.rmse(y=X,S=None).T,axis=0)
#     bw = np.mean(librosa.feature.spectral_bandwidth(y=X,S=None).T,axis=0)
#     features = [mfccs,chroma,mel,contrast,tonnetz,zero_crossing_rate,rmse,bw]

#     embed()
#     return features

# def parse_audio_files(parent_dir,sub_dirs,num_ex,file_ext="*.wav"):
#     features, labels = np.empty((0,196)), np.empty(0)
#     # features, labels = [], []
#     for label, sub_dir in enumerate(sub_dirs):
#         i = 0
#         for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
#             i +=1
#             try:
#               new_features = extract_feature(fn)
#             except Exception as e:
#               print "Error encountered while parsing file: ", fn
#               continue
#             ext_features = np.hstack(new_features)
#             features = np.vstack([features,ext_features])
#             labels = np.append(labels, fn.split('/')[7].split('-')[1])
#             if i > num_ex:
#                 break
#     return np.array(features), np.array(labels, dtype = np.int)


# def extract_feature(file_name):
#     X, sample_rate = librosa.load(file_name)
#     stft = np.abs(librosa.stft(X))
#     mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
#     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
#     mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
#     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
#     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
#     return mfccs,chroma,mel,contrast,tonnetz

# def parse_audio_files(parent_dir,sub_dirs,num_ex,file_ext="*.wav"):
#     features, labels = np.empty((0,193)), np.empty(0)
#     for label, sub_dir in enumerate(sub_dirs):
#         i = 0
#         for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
#             i +=1
#             try:
#               mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
#             except Exception as e:
#               print "Error encountered while parsing file: ", fn
#               continue
#             ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#             features = np.vstack([features,ext_features])
#             labels = np.append(labels, fn.split('/')[7].split('-')[1])
#             if i > num_ex:
#                 break
#     return np.array(features), np.array(labels, dtype = np.int)

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
            # features = np.vstack([features,mel])
            # labels = np.append(labels, fn.split('/')[7].split('-')[1])
            # if i > num_ex:
            #     break
    return np.array(features), np.array(labels, dtype = np.int)

# def one_hot_encode(labels):
#     n_labels = len(labels)
#     n_unique_labels = len(np.unique(labels))
#     one_hot_encode = np.zeros((n_labels,n_unique_labels))
#     one_hot_encode[np.arange(n_labels), labels] = 1
#     return one_hot_encode

parent_dir = '/home/franalli/Documents/UrbanSound8K/audio/'
train_sub_dirs = ["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8"]
val_sub_dirs = ["fold9"]
test_sub_dirs = ["fold10"]

train_features, train_labels = parse_audio_files(parent_dir,train_sub_dirs,1)
# val_features, val_labels = parse_audio_files(parent_dir,val_sub_dirs,10000)
# test_features, test_labels = parse_audio_files(parent_dir,test_sub_dirs,100)

embed()
np.savetxt('/home/franalli/Documents/UrbanSound8K/train',train_features)
np.savetxt('/home/franalli/Documents/UrbanSound8K/val',val_features)
# np.savetxt('/home/franalli/Documents/UrbanSound8K/test',test_features)
np.savetxt('/home/franalli/Documents/UrbanSound8K/y_train',train_labels)
np.savetxt('/home/franalli/Documents/UrbanSound8K/y_val',val_labels)
# np.savetxt('/home/franalli/Documents/UrbanSound8K/y_test',test_labels)