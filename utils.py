"""
Created by: Glenn Kroegel
Date: 20 April, 2019
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
from config import MU, STD

def T(a):
    if torch.is_tensor(a):
        res = a
    else:
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            res = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            res = torch.FloatTensor(a.astype(np.float32))
        else:
            raise NotImplementedError(a.dtype)
    return to_gpu(res)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def scale(x):
    return (x-MU)/STD
    # return (2*x - x.max() - x.min())/(x.max()-x.min())

USE_GPU=False
def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() and USE_GPU else x

def accuracy(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()

def bce_acc(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    bs = input.size(0)
    input = torch.round(input).view(bs,1)
    targs = targs.view(bs,1)
    return (input==targs).float().mean()

def cos_acc(input, targs):
    "Computes correct similarity classification for two embeddings."
    v1, v2 = input
    bs = v1.size(0)
    p = F.cosine_similarity(v1,v2)
    p[p < 0] = 0
    preds = torch.round(p).view(bs,1)
    targs = targs.view(bs,1)
    targs[targs==-1] = 0
    acc = (preds==targs).float().mean()

    # False positive rate in high threshold (hard negatives)
    ixs = (p > 0.8).nonzero().squeeze()
    strong_pairs = torch.round(p[ixs])
    strong_targs = targs[ixs]
    hard_fp_rate = (strong_pairs != strong_targs).float().mean()
    return acc, hard_fp_rate

def triple_accuracy(vecs):
    v1, v2, v3 = vecs
    bs = v1.size(0)
    ap_sim = F.cosine_similarity(v1, v2)
    an_sim = F.cosine_similarity(v1, v3)
    ap_mean = ap_sim.mean()
    ap_min = ap_sim.min()
    an_mean = an_sim.mean()
    return ap_min, ap_mean, an_mean

def fbeta(y_pred, y_true, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `y_pred` and `y_true` in a multi-classification task."
    beta2 = beta**2
    import pdb; pdb.set_trace()
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()