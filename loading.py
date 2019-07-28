"""
created_by: Glenn Kroegel
date: 21 April 2019

description: Create dataloaders to feed for training

"""
import pandas as pd
import numpy as np
from config import TRAIN_INFO, AC_DATA, BATCH_SIZE, TRAIN_SET, CV_SET
from dataset import DatasetProcessor, ProcdDataset1, WeightedDataset, ProcdDataset2, ClassificationDataset

import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader

class SampleLoaderFactory():
    '''Standard dataloaders with regular collating/sampling from padded dataset'''
    def __init__(self):
        self.train_set = ProcdDataset1(pd.read_pickle(TRAIN_SET))
        self.cv_set = ProcdDataset1(pd.read_pickle(CV_SET))
        self.train_sampler = RandomSampler(self.train_set)
        self.cv_sampler = RandomSampler(self.cv_set)

    def gen_loaders(self, batch_size=BATCH_SIZE):
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=self.train_sampler, collate_fn=self.collate_fn)
        self.cv_loader = DataLoader(self.cv_set, batch_size=batch_size, sampler=self.cv_sampler, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return [x, y]

    def save_loaders(self):
        torch.save(self.train_loader, 'train_loader.pt', pickle_protocol=4)
        torch.save(self.cv_loader, 'cv_loader.pt', pickle_protocol=4)

class ClassificationLoaderFactory():
    '''Standard dataloaders with regular collating/sampling from padded dataset'''
    def __init__(self):
        self.train_set = ClassificationDataset(pd.read_pickle(TRAIN_SET))
        self.cv_set = ClassificationDataset(pd.read_pickle(CV_SET))
        self.train_sampler = RandomSampler(self.train_set)
        self.cv_sampler = RandomSampler(self.cv_set)

    def gen_loaders(self, batch_size=BATCH_SIZE):
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=self.train_sampler, collate_fn=self.collate_fn)
        self.cv_loader = DataLoader(self.cv_set, batch_size=batch_size, sampler=self.cv_sampler, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return [x, y]

    def save_loaders(self):
        torch.save(self.train_loader, 'train_loader.pt', pickle_protocol=4)
        torch.save(self.cv_loader, 'cv_loader.pt', pickle_protocol=4)

class ScaledLoaderFactory():
    '''Standard dataloaders with regular collating/sampling from padded dataset'''
    def __init__(self):
        self.train_set = ProcdDataset2(pd.read_pickle(TRAIN_SET))
        self.cv_set = ProcdDataset2(pd.read_pickle(CV_SET))
        self.train_sampler = RandomSampler(self.train_set)
        self.cv_sampler = RandomSampler(self.cv_set)

    def gen_loaders(self, batch_size=BATCH_SIZE):
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=self.train_sampler, collate_fn=self.collate_fn)
        self.cv_loader = DataLoader(self.cv_set, batch_size=batch_size, sampler=self.cv_sampler, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        x_raw = [item[0] for item in batch]
        x_unscaled = [item[1] for item in batch]
        y = [item[2] for item in batch]
        return [x_raw, x_unscaled, y]

    def save_loaders(self):
        torch.save(self.train_loader, 'train_loader.pt', pickle_protocol=4)
        torch.save(self.cv_loader, 'cv_loader.pt', pickle_protocol=4)

class WeightedLoaderFactory():
    '''Standard dataloaders with regular collating/sampling from padded dataset'''
    def __init__(self):
        self.train_set = WeightedDataset(pd.read_pickle(TRAIN_SET))
        self.cv_set = WeightedDataset(pd.read_pickle(CV_SET))
        self.train_sampler = RandomSampler(self.train_set)
        self.cv_sampler = RandomSampler(self.cv_set)

    def gen_loaders(self, batch_size=BATCH_SIZE):
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=self.train_sampler, collate_fn=self.collate_fn)
        self.cv_loader = DataLoader(self.cv_set, batch_size=batch_size, sampler=self.cv_sampler, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        w = [item[2] for item in batch]
        return [x, y, w]

    def save_loaders(self):
        torch.save(self.train_loader, 'train_loader.pt', pickle_protocol=4)
        torch.save(self.cv_loader, 'cv_loader.pt', pickle_protocol=4)