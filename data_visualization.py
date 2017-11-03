#copyright: https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20NN.ipynb

import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed
# %matplotlib inline
plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 200)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    # plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 200)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    # plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 200)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()


def plot_single_example(n,f):
    fig = plt.figure()
    plt.subplot(4,1,1)
    librosa.display.waveplot(np.array(f)[0],sr=22050,x_axis='off')
    plt.title('Sound Wave',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Amplitude [m]',fontsize=20)

    plt.subplot(4,1,2)
    # specgram(np.array(f)[0], Fs=22050)
    S = librosa.feature.melspectrogram(y=np.array(f)[0],power=2.0)
    librosa.display.specshow(S,y_axis='mel') 
    plt.title('Mel Spectrogram',fontsize=20) 
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=20)
    plt.ylabel('Frequency [Hz]',fontsize=20)

    plt.subplot(4,1,3)
    C = librosa.feature.chroma_stft(np.array(f)[0])
    librosa.display.specshow(C,y_axis='chroma')  
    plt.title('Chromagram',fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Pitch Class',fontsize=20)

    plt.subplot(4,1,4)
    D = librosa.logamplitude(np.abs(librosa.stft(f[0]))**2, ref_power=np.max)
    librosa.display.specshow(D,x_axis='time' ,y_axis='log')  
    plt.title('Log Amplitude Spectrogram',fontsize=20)
    plt.xlabel('Time',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Frequency [Hz]',fontsize=20)
    plt.show()

# sound_file_paths = ["57320-0-0-7.wav","24074-1-0-3.wav","15564-2-0-1.wav","31323-3-0-1.wav","46669-4-0-35.wav",
#                    "89948-5-0-0.wav","40722-8-0-4.wav","103074-7-3-2.wav","106905-8-0-0.wav","108041-9-0-4.wav"]
# sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
#                "gun shot","jackhammer","siren","street music"]

sound_file_paths = ["24074-1-0-3.wav"]
sound_names = ["Car Horn"]


# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music
folder_path = '/home/franalli/Documents/UrbanSound8K/audio/fold1/'
sound_file_paths = [folder_path+item for item in sound_file_paths]

raw_sounds = load_sound_files(sound_file_paths)
# plot_waves(sound_names,raw_sounds)
# plot_specgram(sound_names,raw_sounds)
# plot_log_power_specgram(sound_names,raw_sounds)
plot_single_example(sound_names,raw_sounds)