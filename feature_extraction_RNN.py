#copyright: https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20NN.ipynb

import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed
import cPickle as pickle


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

# frames are the time steps, bands are the dimension of each timestep
def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 50, frames = 100):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            label = fn.split('/')[2].split('-')[1]
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)         
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)
    return np.array(features), np.array(labels,dtype = np.int)


parent_dir = 'audio/'
train_sub_dirs = ["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8"]
val_sub_dirs = ["fold9"]
test_sub_dirs = ["fold10"]

train_features, train_labels = extract_features(parent_dir,train_sub_dirs)
val_features, val_labels = extract_features(parent_dir,val_sub_dirs)
# test_features, test_labels = parse_audio_files(parent_dir,test_sub_dirs,100)

pickle.dump( [train_features,val_features,train_labels,val_labels], open( "RNN.p", "wb" ) )