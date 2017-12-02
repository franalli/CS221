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


[train_data,val,y_train_data,y_val] = pickle.load( open( "CNN.p", "rb" ) )
# embed()
# train_data = np.transpose(train_data,[0,1,2,3])
# val = np.transpose(val,[0,1,2,3])

N_t = 8000
N_v = 400
W = 64
D = 64
C = 2
# train_data = np.random.rand(N_t,C,D,W)
# val = np.random.rand(N_v,C,D,W)
# y_train_data = np.random.randint(10, size=N_t)
# y_val = np.random.randint(10, size=N_v)

D_out = 10
num_epochs = 200
learning_rate = 0.001
batch_size = 100
dropout = 0.85
dtype_data = torch.FloatTensor
dtype_labels = torch.LongTensor
# dtype_data = torch.cuda.FloatTensor
# dtype_labels = torch.cuda.LongTensor


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

# [conv - BN - relu ]*N -2x2 max pool - affine - relu - affine - softmax
class CNN(nn.Module):
    def __init__(self,W,D,C,D_out,dropout,batch_size):
        super(CNN, self).__init__()
        self.training = True

        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(4608,10)
        # self.bn5 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(p=dropout)

        # self.score = nn.Linear(1000, D_out)

        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.conv4.weight)
        nn.init.xavier_uniform(self.linear1.weight)

    def forward(self, x):
        self.dropout.training = self.training
        self.bn1.training = self.training
        self.bn2.training = self.training
        self.bn3.training = self.training
        self.bn4.training = self.training
        # self.bn5.training = self.training

        h1 = self.relu(self.bn1(self.conv1(x)))
        h2 = self.relu(self.bn2(self.conv2(h1)))
        h3 = self.relu(self.bn3(self.conv3(h2)))
        h4 = self.relu(self.bn4(self.conv4(h3)))
        h5 = self.maxpool(h4)
        h5 = h5.view(int(h5.size()[0]),4608)

        return self.linear1(h5)
        # h6 = self.dropout(self.relu(self.bn5(self.linear1(h5))))
        # return self.score(h6)


print 'Building the model'
NN = CNN(W,D,C,D_out,dropout,batch_size)
loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate, weight_decay=0.001)


print 'Starting Training'
for epoch in range(num_epochs):
    NN.training = True
    for i,batch in enumerate(getMinibatches([train_data,y_train_data],batch_size)):
        # print (i)
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


    train_accuracy = np.mean(y_train_data[training_subset_ix] == train_indices.data.numpy())
    train_loss = loss_fn(NN(train_set),y_train_var).data[0]
    


    _, val_indices = torch.max(NN(val_full),dim=1)
    val_accuracy = np.mean(y_val == val_indices.data.numpy())
    val_loss = loss_fn(NN(val_full),y_val_var).data[0]
    
    print 'epoch: {}, train_loss: {}, val_loss: {}, train_accuracy: {}, val_accuracy: {}'.format(epoch,train_loss,val_loss,train_accuracy,val_accuracy)
    # print 'epoch: {}, val_loss: {}, val_accuracy: {}'.format(epoch,val_loss,val_accuracy)

