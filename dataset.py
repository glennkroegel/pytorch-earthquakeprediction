import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm

from config import TRAIN_INFO, AC_DATA, MU, STD, CV_SIZE
from utils import T

class DatasetProcessor(torch.utils.data.Dataset):
    def __init__(self):
        self.train_info = pd.read_csv(TRAIN_INFO)
        self.ac_data = self.scale(np.load(AC_DATA)['acoustic_data'])
        self.run()

    def __getitem__(self, i):
        x = self.seqs[i].astype(np.float32)
        y = self.ttfs[i].astype(np.float32)
        x = T(x)
        y = T(y)
        return (x, y)

    def get_quake_period(self, i):
        index_start, chunk_length = self.train_info['index_start'][i], self.train_info['chunk_length'][i]
        t_start, t_end = self.train_info['t_start'][i], self.train_info['t_end'][i]
        ac_data_period = self.ac_data[ index_start : index_start + chunk_length ]
        ttf_data_period = np.linspace(t_start, t_end, chunk_length, dtype=np.float32)
        return ac_data_period, ttf_data_period

    def gen_seqs(self):
        # TODO: track period idxs for equal cv sampling (of ttfs too)
        self.seqs = []
        self.ttfs = []
        y_periods = []
        for index, period in tqdm(self.train_info.iterrows()):
            ix_start = int(period['index_start'])
            chunk_length = int(period['chunk_length'])
            t_start = period['t_start']
            t_end = period['t_end']
            y_period = index
            period_data = self.ac_data[ix_start: ix_start+chunk_length]
            period_ttf = np.linspace(t_start, t_end, chunk_length, dtype=np.float32)
            split_size = 5000
            seq = np.array_split(period_data, split_size)
            ttf = np.array_split(period_ttf, split_size)
            ttf = [y[-1] for y in ttf]
            self.seqs.extend(seq)
            self.ttfs.extend(ttf)
        # self.gen_sets()

    def get_idxs(self):
        idxs = list(np.arange(len(self.seqs)))
        samples = int(len(idxs)*CV_SIZE)
        cv_idxs = random.sample(idxs, samples)
        train_idxs = [x for x in idxs if x not in cv_idxs]
        return train_idxs, cv_idxs

    def gen_sets(self):
        train_idxs, cv_idxs = self.get_idxs()
        train_set = [(T(self.seqs[a]), T(self.ttfs[a])) for a in train_idxs]
        cv_set = [(T(self.seqs[a]), T(self.ttfs[a])) for a in cv_idxs]
        return train_set, cv_set

    def run(self):
        self.gen_seqs()
        train_set, cv_set = self.gen_sets()
        pd.to_pickle(train_set, 'train_set.pkl')
        pd.to_pickle(cv_set, 'cv_set.pkl')

    def gen_noise_samples(self):
        pass

    def scale(self, x, u=MU, std=STD):
        return (x-u)/std

    def __len__(self):
        return len(self.train_info)

class ProcdDataset1(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = [a[0] for a in data]
        self.y = torch.FloatTensor([a[1] for a in data])

    def __getitem__(self, i):
        return [self.x[i], self.y[i]]

    def __len__(self):
        return len(self.y)

class ProcdDataset2(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = [a[0] for a in data]
        self.xs = [a[1] for a in data]
        self.y = torch.FloatTensor([a[2] for a in data])

    def __getitem__(self, i):
        return [self.x[i], self.xs[i], self.y[i]]

    def __len__(self):
        return len(self.y)

class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = [a[0] for a in data]
        self.y = torch.FloatTensor([a[1] for a in data])
        self.w = torch.tensor(pd.read_pickle('l1_wgts.pkl'))

    def __getitem__(self, i):
        return [self.x[i], self.y[i], self.w[i]]

    def __len__(self):
        return len(self.y)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = [a[0] for a in data]
        self.y = torch.LongTensor([a[1] for a in data])
        # self.w = torch.tensor(pd.read_pickle('l1_wgts.pkl'))

    def __getitem__(self, i):
        # return [self.x[i], self.y[i], self.w[i]]
        return [self.x[i], self.y[i]]

    def __len__(self):
        return len(self.y)