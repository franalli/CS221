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

def extract_features(parent_dir,sub_dirs,numex,file_ext="*.wav",bands = 64, frames = 64):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    features = np.zeros((1,2,64,64))
    for l, sub_dir in enumerate(sub_dirs):
        i = 0
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            i+=1
            sound_clip,s = librosa.load(fn)
            label = fn.split('/')[2].split('-')[1]
            # for (start,end) in windows(sound_clip,window_size):
            # if(len(sound_clip[start:end]) == window_size):
            # print start,end     
            # signal = sound_clip[start:end]
            # melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            hop = len(sound_clip)/(frames-1)
            if len(sound_clip)/hop != (frames-1):
                continue

            melspec = librosa.feature.melspectrogram(sound_clip,n_mels=bands,hop_length=hop)
            logspec = librosa.logamplitude(melspec)
            delta = librosa.feature.delta(logspec)
            feature = np.expand_dims(np.concatenate((np.expand_dims(logspec,0),np.expand_dims(delta,0)),axis=0),axis=0)
            try:
                features = np.vstack((features,feature))
            except:
                embed()

            labels.append(label)

            if i > numex:
                break

    return features[1:], np.array(labels,dtype = np.int)


parent_dir = 'audio/'
train_sub_dirs = ["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8"]
val_sub_dirs = ["fold9"]
test_sub_dirs = ["fold10"]

train_features, train_labels = extract_features(parent_dir,train_sub_dirs,float('inf'))
val_features, val_labels = extract_features(parent_dir,val_sub_dirs,float('inf'))
# test_features, test_labels = parse_audio_files(parent_dir,test_sub_dirs,100)

# N,W,H,C = np.shape(train_features)
# train_features = np.reshape(train_features,[N,W*H*C])
# N,W,H,C = np.shape(val_features)
# val_features = np.reshape(val_features,[N,W*H*C])
pickle.dump( [train_features,val_features,train_labels,val_labels], open( "CNN.p", "wb" ) )

# np.savetxt('train',train_features)
# np.savetxt('val',val_features)
# np.savetxt('/home/franalli/Documents/UrbanSound8K/test',test_features)
# np.savetxt('y_train',train_labels)
# np.savetxt('y_val',val_labels)
# np.savetxt('/home/franalli/Documents/UrbanSound8K/y_test',test_labels)