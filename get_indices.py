# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:26:48 2020

@author: Tim
"""

import pandas as pd
import random
import pickle

df = pd.read_csv('sequence_data.tsv',sep='\t')

print(len(df))

df.drop_duplicates(subset = ['seqnames','start','strand'], inplace=True)
df = df[df['status'] < 2]
all_ind = list(df.index)
totals = len(all_ind)
print(totals)

pos_ind = list(df[df['status'] == 1].index)
neg_ind = list(df[df['status'] == 0].index)

print(len(neg_ind))
print(len(pos_ind))

with open('all','wb') as f:
    pickle.dump(all_ind,f)
with open('pos_neg','wb') as f:
    pickle.dump([pos_ind,neg_ind],f)

