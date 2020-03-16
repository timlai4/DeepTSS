# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 00:34:45 2020

@author: Tim
"""

import pandas as pd
import sklearn.metrics

df = pd.read_csv('prediction_reports.tsv', sep = '\t')
promoters = df[df['simple_annotations'] == 'promoter']
len(promoters)

#exons = df[df['simple_annotations'] == 'exon']
#len(exons)

test = promoters[promoters['score'] > 50]
test.status = [1 for _ in range(len(test))]

print(sklearn.metrics.mean_absolute_error(test.status, test.KerasLogitProb))
print(sklearn.metrics.mean_absolute_error(test.status, test.LogitProb))
print(sklearn.metrics.mean_absolute_error(test.status, test.DNNProb))