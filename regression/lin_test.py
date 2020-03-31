import torch
import torch.nn as nn
#import torch.optim as optim
import pandas as pd
import numpy as np
import pickle

class Linear(nn.Module):

    def __init__(self, input_size):
        super(Linear, self).__init__()        
        self.out = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.out(x)
        return x

# Load and process the data
testdir = 'data/yeast_diamide/'
df = pd.read_csv(testdir + 'unique_to_diamide.tsv',sep='\t')
with open(testdir + 'test-indices','rb') as f:
    all, tss, soft = pickle.load(f)
df = df.iloc[all]
score = np.array(np.log(df['score']))

gen = np.load(testdir + 'genomic.npy')
gen = np.reshape(gen[tss], (gen[tss].shape[0], gen[tss].shape[1], gen[tss].shape[2]))
signal = np.load(testdir + 'signal.npy')
signal = np.reshape(signal[tss], (signal[tss].shape[0], signal[tss].shape[1], signal[tss].shape[2]))
clip = np.load(testdir + 'softclipped.npy')
clip = np.reshape(clip[soft], (clip[soft].shape[0], clip[soft].shape[1], clip[soft].shape[2]))
seq_shape = np.load(testdir + 'shape.npy')
seq_shape = np.reshape(seq_shape[tss], (seq_shape[tss].shape[0], seq_shape[tss].shape[1], seq_shape[tss].shape[2]))

gen = torch.from_numpy(gen)
signal = torch.from_numpy(signal)
clip = torch.from_numpy(clip)
seq_shape = torch.from_numpy(seq_shape)
score = torch.from_numpy(score).view(-1,1).type(torch.float)

gen = torch.flatten(gen, start_dim = 1)
signal = torch.flatten(signal, start_dim = 1)
clip = torch.flatten(clip, start_dim = 1)
seq_shape = torch.flatten(seq_shape, start_dim = 1)
inputs = torch.cat((gen, signal, clip, seq_shape), 1)

input_size = inputs.size(1)
lin = Linear(input_size)
PATH = 'lin.pth'
lin.load_state_dict(torch.load(PATH))
lin = lin.float()
with torch.no_grad():
    outputs = lin(inputs.float())
#    pred = torch.round(outputs)
df['Lin_Pred'] = outputs.numpy()
df.to_csv('seq_pred.tsv',sep = '\t', index = False)
