import glob
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from IPython import embed
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import torch
import random
from torch.autograd import Variable
import torch.nn.functional
import torch.nn as nn
import cPickle as pickle


[train_data,val,y_train_data,y_val] = pickle.load( open( "RNN.p", "rb" ) )
train_data = np.transpose(train_data,[0,1,2])
val = np.transpose(val,[0,1,2])
embed()

N_t = 8000
N_v = 400
W = 100
D = 50
# train_data = np.random.rand(N_t,W,D)
# val = np.random.rand(N_v,W,D)
# y_train_data = np.random.randint(10, size=N_t)
# y_val = np.random.randint(10, size=N_v)

D_out = 10
num_epochs = 200
learning_rate = 0.001
batch_size = 100
dropout = 0.85
# dtype_data = torch.FloatTensor
# dtype_labels = torch.LongTensor
dtype_data = torch.cuda.FloatTensor
dtype_labels = torch.cuda.LongTensor


val_full = Variable(torch.from_numpy(val).type(dtype_data),requires_grad=False)
y_val_var = Variable(torch.from_numpy(y_val).type(dtype_labels),requires_grad=False)


def getMinibatches(data, batch_size, shuffle=True):
    num_data = len(data[0])
    sounds,labels = data
    indices = np.arange(num_data)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, num_data, batch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
        yield [np.array(sounds[minibatch_indices]),np.array(labels[minibatch_indices])]

class RNN(nn.Module):
    def __init__(self,W,D,D_out,dropout,batch_size):
        super(RNN, self).__init__()
        self.training = True

        self.lstm = nn.LSTM(D,100,W,bias=True,batch_first=True,dropout=dropout,bidirectional=False)
        self.score = nn.Linear(100, D_out)

        nn.init.xavier_uniform(self.score.weight)

    def forward(self, x):
        out,hn = self.lstm(x)
        h = out[:,-1,:]
        return self.score(h)


print 'Building the model'
NN = RNN(W,D,D_out,dropout,batch_size).cuda()
loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate, weight_decay=0.001)


print 'Starting Training'
for epoch in range(num_epochs):
    NN.training = True
    for i,batch in enumerate(getMinibatches([train_data,y_train_data],batch_size)):
        print (i)
        train = Variable(torch.from_numpy(batch[0]).type(dtype_data), requires_grad=True)
        y_train = Variable(torch.from_numpy(batch[1]).type(dtype_labels),requires_grad=False)
        y_pred_train = NN(train)


        loss = loss_fn(y_pred_train,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # End of epoch evaluation metrics
    NN.training = False

    training_subset_ix = np.random.randint(np.shape(train_data)[0],size=400)
    train_set = train_data[training_subset_ix]
    train_set = Variable(torch.from_numpy(train_set).type(dtype_data),requires_grad=False)
    y_train_var = Variable(torch.from_numpy(y_train_data[training_subset_ix]).type(dtype_labels),requires_grad=False)

    _, train_indices = torch.max(NN(train_set),dim=1)

    train_indices = train_indices.cpu()
    train_accuracy = np.mean(y_train_data[training_subset_ix] == train_indices.data.numpy())
    train_loss = loss_fn(NN(train_set),y_train_var).data[0]
    


    _, val_indices = torch.max(NN(val_full),dim=1)
    val_indices = val_indices.cpu()
    val_accuracy = np.mean(y_val == val_indices.data.numpy())
    val_loss = loss_fn(NN(val_full),y_val_var).data[0]
    
    print 'epoch: {}, train_loss: {}, val_loss: {}, train_accuracy: {}, val_accuracy: {}'.format(epoch,train_loss,val_loss,train_accuracy,val_accuracy)

