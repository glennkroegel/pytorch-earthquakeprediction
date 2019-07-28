'''
created_by: Glenn Kroegel
date: 21 April 2019

description: Neural networks

https://www.kaggle.com/jsaguiar/seismic-data-exploration

TODO: linear detector first layer
TODO: experiment with bias

1) Stateful rnn, reset on new experiment (12 total)
2) filter by high values, use ixs of high values (scaled 0-1) for dense input
3) 2D conv on image transformation of series
4) RNN takes 150k input but outputs a y value for every sample (so will have differnet x.size and y.size output)
5) Dense network on the polar coords (probaby have to max pool to usable size)
6) binned classifier as base and several models for specific ttf value
'''
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from utils import T, accuracy, count_parameters

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchnlp.nn import Attention
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm

from config import NUM_EPOCHS, BATCH_SIZE

class Dense(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.bn = nn.BatchNorm1d(out_size)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.act(self.fc(x))
        # x = self.act(self.fc(x))
        return x

class Conv(nn.Module):
    def __init__(self, in_c, out_c, ks, stride=1, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, bias=bias)
        self.bn = nn.BatchNorm1d(self.conv.out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        # x = self.act(self.conv(x))
        return x

class LSTMEncoder(nn.Module):
    """ LSTM encoding of one sequence"""
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.input_dim = 1
        self.hidden_dim = 16
        self.nl = 1
        self.bidir = True
        self.direction = 1
        if self.bidir:
            self.direction = 2

        # Layers
        # self.pool0 = nn.MaxPool1d(10, 1)
        self.pool0 = nn.AdaptiveMaxPool1d(400)
        self.rnn = nn.LSTM(
            input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.nl,
            bidirectional=self.bidir, dropout=0.1, batch_first=True)
        self.fc1 = Dense(32, 32)
        self.fc2 = Dense(32, 16)
        self.fc3 = Dense(16, 8)
        self.act = nn.ELU()
        self.fc_out = nn.Linear(16, 1, bias=True)

    def init_hidden(self, batch_size):
        """ Re-initializes the hidden state, cell state, and the forget gate bias of the network. """
        h0 = Variable(torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim))#.to(self.device)
        c0 = Variable(torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim))#.to(self.device)
        return h0, c0

    def forward(self, batch_size, input_data):
        """ Performs a forward pass through the network. """
        x = [self.pool0(a.view(1,1,-1)).squeeze(1).transpose_(0,1) for a in input_data]
        # x = torch.stack(x, dim=-1)
        # x[x>2] = 2
        lens = [a.size(-1) for a in x]
        indices = np.argsort(lens)[::-1].tolist()
        rev_ind = [indices.index(i) for i in range(len(indices))]
        x = [x[i] for i in indices]
        # import pdb; pdb.set_trace()
        x = pad_sequence(x, batch_first=True)
        input_lengths = [lens[i] for i in indices]
        sl = x.size(1)
        output = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        hidden, cell = self.init_hidden(batch_size)
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[rev_ind, :].contiguous()
        hidden = hidden.transpose_(0,1)[rev_ind, :, :].contiguous()
        # hidden = self.act(self.proj(hidden.view(batch_size, -1)))
        x = self.fc2(hidden.view(batch_size, -1))
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.act(self.fc_out(x))
        return x

class LSTMSeq(nn.Module):
    """ LSTM encoding of one sequence"""
    def __init__(self):
        super(LSTMSeq, self).__init__()
        self.input_dim = 1
        self.hidden_dim = 16
        self.nl = 2
        self.bidir = True
        self.direction = 1
        if self.bidir:
            self.direction = 2

        # Layers
        # self.pool0 = nn.MaxPool1d(10, 1)
        self.pool0 = nn.AdaptiveMaxPool1d(100)
        self.rnn = nn.LSTM(
            input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.nl,
            bidirectional=self.bidir, dropout=0.1, batch_first=True)
        self.fc1 = Dense(64, 32)
        self.fc2 = Dense(128, 64)
        self.fc3 = Dense(64, 16)
        self.act = nn.Tanh()
        self.fc_out = nn.Linear(16, 1, bias=True)

    def init_hidden(self, batch_size):
        """ Re-initializes the hidden state, cell state, and the forget gate bias of the network. """
        h0 = Variable(torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim))#.to(self.device)
        c0 = Variable(torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim))#.to(self.device)
        return h0, c0

    def forward(self, batch_size, input_data):
        """ Performs a forward pass through the network. """
        x = [self.pool0(a.view(1,1,-1)).squeeze(1).transpose_(0,1) for a in input_data]
        x = torch.stack(x, dim=0)
        hidden, cell = self.init_hidden(batch_size)
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        hidden = hidden.transpose_(0,1).contiguous()
        # hidden = self.act(self.proj(hidden.view(batch_size, -1)))
        x = self.fc1(hidden.view(batch_size, -1))
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.act(self.fc_out(x))
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.pool0 = nn.AdaptiveMaxPool1d(100)
        self.conv1 = Conv(1, 10, 3, stride=1, dilation=1)
        self.pool1 = nn.AdaptiveMaxPool1d(100)
        self.conv2 = Conv(10, 20, 2)
        self.pool2 = nn.AdaptiveMaxPool1d(10)
        self.conv3 = Conv(20, 10, 1)
        self.pool3 = nn.AdaptiveMaxPool1d(5)
        self.fc1 = Dense(400, 400)
        self.fc2 = Dense(400, 100)
        self.fc3 = Dense(50,32)
        self.fc4 = Dense(32, 16)
        self.fc5 = Dense(32, 8)
        self.act = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(8, 1, bias=False)

    def forward(self, x):
        bs = len(x)
        x = [self.pool0(a.view(1,1,-1)) for a in x]
        x = torch.stack(x, dim=0).squeeze(1)
        # sl = x.size(2)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = x.view(bs, -1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
        # # x = self.fc5(x)
        # x = self.act(self.fc_out(x))
        return x

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__(size=100)
        self.max_pool = nn.AdaptiveMaxPool1d(100)
        self.avg_pool = nn.AdaptiveAvgPool1d(50)
        self.fc1 = Dense(10000, 100)
        self.fc2 = Dense(100, 64)
        self.fc3 = Dense(64, 32)
        # self.fc4 = Dense(32, 8)
        self.act = nn.Tanh()
        # self.fc_out = nn.Linear(8, 1, bias=True)

    def forward(self, x):
        bs = len(x)
        maxs = [self.max_pool(a.view(1,1,-1)) for a in x]
        avgs = [self.avg_pool(a.view(1,1,-1)) for a in x]
        maxs = torch.stack(maxs, dim=0).squeeze(1)
        avgs = torch.stack(avgs, dim=0).squeeze(1)
        x = torch.cat([maxs, avgs], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
        # x = self.act(self.fc_out(x))
        return x

class PolarNet(nn.Module):
    def __init__(self, size=10):
        super(PolarNet, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(size, return_indices=True)
        self.fc1 = Dense(2*size, 100)
        self.fc2 = Dense(100, 64)
        self.fc3 = Dense(64, 32)
        self.fc4 = Dense(32, 32)
        self.fc5 = Dense(32, 32)
        # self.act = nn.Tanh()
        # self.fc_out = nn.Linear(32, 1, bias=True)

    def forward(self, x):
        bs = len(x)
        maxs = [self.max_pool(a.view(1,-1,1)) for a in x]
        xs = torch.stack([x[0].squeeze() for x in maxs], dim=0)
        xs = torch.cos(xs)
        ixs = torch.stack([x[1].float().squeeze()/150000 for x in maxs], dim=0)
        x = torch.cat([ixs, xs], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        # x = self.fc_out(x)
        return x

class JointNetwork(nn.Module):
    def __init__(self):
        super(JointNetwork, self).__init__()
        self.net1 = PolarNet(100)
        self.net2 = PolarNet(100)
        # self.net3 = LSTMSeq()

        self.fc1 = Dense(32*2, 32)
        self.fc2 = Dense(32, 16)
        self.fc3 = Dense(16,8)
        self.out = nn.Linear(8,1)
        self.act = nn.Tanh()

    def forward(self, x, x2):
        bs = len(x)
        x1 = self.net1(x).view(bs,-1)
        x2 = self.net2(x2).view(bs,-1)
        # x3 = self.net3(bs, x).view(bs,-1)
        x = torch.cat([x1,x2], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x

    def get_ttf(self, x):
        with torch.no_grad():
            return self.forward(x).numpy()

class SingleNetwork(nn.Module):
    def __init__(self):
        super(SingleNetwork, self).__init__()
        self.net1 = PolarNet()
        self.fc = Dense(32,8)
        self.out = nn.Linear(8,6)
        self.act = nn.Tanh()

    def forward(self, x):
        bs = len(x)
        x = self.net1(x)#.view(bs,-1)
        x = self.fc(x)
        x = self.out(x)
        x = x.view(bs,-1)
        return x

    def get_ttf(self, x):
        with torch.no_grad():
            return self.forward(x).numpy()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, outputs, targets):
        targets = targets.view(-1,1)
        logpt = F.log_softmax(outputs, dim=-1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=outputs.data.type():
                self.alpha = self.alpha.type_as(outputs.data)
            at = self.alpha.gather(0,targets.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class Learner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = SingleNetwork()
        self.alpha = list(pd.read_pickle('class_weights.pkl').values())
        self.criterion = FocalLoss(gamma=2, alpha=self.alpha)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=5e-5)
        self.epochs = epochs

        self.train_loader = torch.load('train_loader.pt')
        self.cv_loader = torch.load('cv_loader.pt')

        self.best_loss = 1e3
        print(count_parameters(self.model))

    def train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        train_loss = 0
        train_acc = 0
        for _, data in enumerate(train_loader):
            seq, targets = data
            targets = torch.stack(targets)
            # weight = torch.stack(weight)
            output = model(seq)
            loss = criterion(output, targets)
            train_loss += loss.item()
            train_acc += accuracy(output, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clip_grad_norm_(self.model.parameters(), 0.25)
        train_loss = train_loss/len(train_loader)
        train_acc = train_acc/len(train_loader)
        return train_loss, train_acc

    def step(self):
        '''Actual training loop.'''
        for epoch in tqdm(range(self.epochs)):
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()[0]
            epoch_loss, train_acc = self.train(self.train_loader, self.model, self.criterion, self.optimizer, epoch)
            # cross validation
            total_val_loss = 0
            with torch.no_grad():
                for _, data in enumerate(self.cv_loader):
                    seq, targets = data
                    targets = torch.stack(targets)
                    # weight = torch.stack(weight)
                    self.model.eval()
                    val_outputs = self.model(seq)
                    v1 = val_outputs
                    val_loss = self.criterion(v1, targets)
                    total_val_loss += val_loss.item()
                epoch_val_loss = total_val_loss/len(self.cv_loader)
                if epoch % 1 == 0:
                    self.status(epoch, epoch_loss, train_acc, epoch_val_loss, lr)
                if epoch_val_loss < self.best_loss:
                    print('dumping model...')
                    path = 'model' + '.pt'
                    torch.save(self.model, path)
                    self.best_loss = epoch_val_loss

    def status(self, epoch, acc, epoch_loss, epoch_val_loss, lr):
        print('epoch {0}/{1}:\n train_loss: {2} acc: {3} val_loss: {4} learning_rate: {5}'
        .format(epoch, self.epochs, acc, epoch_loss, epoch_val_loss, lr))


if __name__ == "__main__":
    try:
        Learner().step()
    except KeyboardInterrupt:
        pass