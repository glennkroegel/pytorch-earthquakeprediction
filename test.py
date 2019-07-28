'''
created_by: Glenn Kroegel
date: 11 May 2019

description: process test results

'''
import pandas as pd
import numpy as np
import json
import torch
import glob
import os
from tqdm import tqdm
from utils import scale, T
from model import *
from config import MODEL_FILE

model = torch.load(MODEL_FILE)
model.eval()

file_list = glob.glob('test_data/*.csv')
res = {}
with torch.no_grad():
    for f in tqdm(file_list):
        seg = pd.read_csv(f).values.reshape(1,-1)
        # seg = scale(seg)
        x1 = T(seg.astype(np.float32))
        x2 = T(scale(seg))
        # x = torch.FloatTensor(seg)
        ttf = float(model.forward(x1,x2).numpy())
        k = f.split('.')[0].split('/')[1]
        res[k] = ttf
df = pd.DataFrame.from_dict(res, orient='index')
df.reset_index(inplace=True)
df.columns = ['seg_id', 'time_to_failure']
df.to_csv('results.csv', index=False)

print(df['time_to_failure'].describe([0.05,0.1,0.25,0.5,0.75,0.9,0.95]))
