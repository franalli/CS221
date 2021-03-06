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

print 'Loading Dataset'
train_data = np.loadtxt('train')
y_train_data = np.loadtxt('y_train')
val = np.loadtxt('val')
y_val = np.loadtxt('y_val')

pca = PCA(n_components=185)
train_data = pca.fit_transform(train_data)
val = pca.transform(val)

N,D = np.shape(train_data)
print 'N:',N,'D:',D
H1 = 200
H2 = 200
H3 = 200
H4 = 200
D_out = 10
num_epochs = 5
learning_rate = 0.0001
batch_size = 20
dropout = 0.65

dtype = torch.FloatTensor # Comment this out to run on GPU
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

val_full = Variable(torch.from_numpy(val).type(dtype),requires_grad=False)
train_full = Variable(torch.from_numpy(train_data).type(dtype),requires_grad=False)
y_val_var = Variable(torch.from_numpy(y_val).type(torch.LongTensor),requires_grad=False)
y_train_var = Variable(torch.from_numpy(y_train_data).type(torch.LongTensor),requires_grad=False)


def getMinibatches(data, batch_size, shuffle=True):
    num_data = len(data[0])
    sounds,labels = data
    indices = np.arange(num_data)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, num_data, batch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
        yield [np.array(sounds[minibatch_indices]),np.array(labels[minibatch_indices])]


class FourLayerNet(nn.Module):
    def __init__(self,D,H1,H2,H3,H4,D_out):
        super(FourLayerNet, self).__init__()
        self.training = True

        self.linear1 = nn.Linear(D, H1)
        self.bn1 = nn.BatchNorm1d(H1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.linear2 = nn.Linear(H1,H2)
        self.bn2 = nn.BatchNorm1d(H2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.linear3 = nn.Linear(H2,H3)
        self.bn3 = nn.BatchNorm1d(H3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)

        self.linear4 = nn.Linear(H3,H4)
        self.bn4 = nn.BatchNorm1d(H4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=dropout)

        self.score = nn.Linear(H4, D_out)

        nn.init.xavier_uniform(self.linear1.weight)
        nn.init.xavier_uniform(self.linear2.weight)
        nn.init.xavier_uniform(self.linear3.weight)
        nn.init.xavier_uniform(self.linear4.weight)

    def forward(self, x):
        self.dropout1.training = self.training
        self.dropout2.training = self.training
        self.dropout3.training = self.training
        self.dropout4.training = self.training
        self.bn1.training = self.training
        self.bn2.training = self.training
        self.bn3.training = self.training
        self.bn4.training = self.training

        h1 = self.dropout1(self.relu1(self.bn1(self.linear1(x))))
        h2 = self.dropout2(self.relu2(self.bn2(self.linear2(h1))))
        h3 = self.dropout3(self.relu3(self.bn3(self.linear3(h2))))
        h4 = self.dropout4(self.relu4(self.bn4(self.linear4(h3))))
        return self.score(h4)


print 'Building the model'
NN = FourLayerNet(D,H1,H2,H3,H4,D_out)
loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate,weight_decay =0.001)

E = range(num_epochs)
T = []
V = []
print 'Starting Training'
for epoch in range(num_epochs):
    NN.training = True
    for i,batch in enumerate(getMinibatches([train_data,y_train_data],batch_size)):
        train = Variable(torch.from_numpy(batch[0]).type(dtype), requires_grad=True)
        y_train = Variable(torch.from_numpy(batch[1]).type(torch.LongTensor),requires_grad=False)
        y_pred_train = NN(train)


        loss = loss_fn(y_pred_train,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # End of epoch evaluation metrics
    NN.training = False
    _, val_indices = torch.max(NN(val_full),dim=1)
    _, train_indices = torch.max(NN(train_full),dim=1)
    val_accuracy = np.mean(y_val == val_indices.data.numpy())
    train_accuracy = np.mean(y_train_data == train_indices.data.numpy())
    train_loss = loss_fn(NN(train_full),y_train_var).data[0]
    val_loss = loss_fn(NN(val_full),y_val_var).data[0]
    T.append(train_loss)
    V.append(val_loss)
    
    print 'epoch: {}, train_loss: {}, val_loss: {}, train_accuracy: {}, val_accuracy: {}'.format(epoch,train_loss,val_loss,train_accuracy,val_accuracy)

# embed()

Tplot, = plt.plot(E,T,linewidth=3)
Vplot, = plt.plot(E,V,linewidth=3)
plt.title('Feed Forward Network Learning Curves',fontsize=20)
plt.xlabel('Epoch',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Softmax Loss',fontsize=20)
plt.legend([Tplot,Vplot],['Training','Validation'],fontsize=20)
plt.show()