import torch
import torch.nn as nn
#import torch.optim as optim
import pandas as pd
import numpy as np
import pickle

class Logit(nn.Module):

    def __init__(self, input_size):
        super(Logit, self).__init__()
        self.out = nn.Linear(input_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.out(x))
        return x

# Load and process the data
df = pd.read_csv('sequence_data.tsv',sep='\t')
test = df['status'] < 2
df = df[test]
status = df.status
status = np.array(status)

#pos = 6931
#neg = 101232
#total = pos + neg

gen = np.load('genomic.npy')
gen = np.reshape(gen[test], (gen[test].shape[0], gen[test].shape[1], gen[test].shape[2]))
signal = np.load('signal.npy')
signal = np.reshape(signal[test], (signal[test].shape[0], signal[test].shape[1], signal[test].shape[2]))
clip = np.load('softclipped.npy')
clip = np.reshape(clip[test], (clip[test].shape[0], clip[test].shape[1], clip[test].shape[2]))
seq_shape = np.load('shape.npy')
seq_shape = np.reshape(seq_shape[test], (seq_shape[test].shape[0], seq_shape.shape[1], seq_shape.shape[2]))

gen = torch.from_numpy(gen)
signal = torch.from_numpy(signal)
clip = torch.from_numpy(clip)
seq_shape = torch.from_numpy(seq_shape)
status = torch.from_numpy(status).view(-1,1).type(torch.float)
#status = status.view(-1,1)
#print(status.type())

gen = torch.flatten(gen, start_dim = 1)
signal = torch.flatten(signal, start_dim = 1)
clip = torch.flatten(clip, start_dim = 1)
seq_shape = torch.flatten(seq_shape, start_dim = 1)
inputs = torch.cat((gen, signal, clip, seq_shape), 1)

input_size = inputs.size(1)
log = Logit(input_size)
PATH = 'logit.pth'
log.load_state_dict(torch.load(PATH))
log = log.float()
with torch.no_grad():
    outputs = log(inputs.float())

