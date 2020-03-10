import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(input_size, 2000)
        self.dense2 = nn.Linear(2000, 2000)
        self.out = nn.Linear(2000, 1)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense2(x))
        x = torch.sigmoid(self.out(x))
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and process the data
df = pd.read_csv('sequence_data.tsv',sep='\t')
#with open('all','rb') as f:
#    test = pickle.load(f)
test = df['status'] < 2
status = df[test].status
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

gen = torch.flatten(gen, start_dim = 1)
signal = torch.flatten(signal, start_dim = 1)
clip = torch.flatten(clip, start_dim = 1)
seq_shape = torch.flatten(seq_shape, start_dim = 1)
inputs = torch.cat((gen, signal, clip, seq_shape), 1)

input_size = inputs.size(1)
net = Net(input_size)
net = net.float()
PATH = 'dnn.pth'
net.load_state_dict(torch.load(PATH))
net = net.float()
with torch.no_grad():
    outputs = net(inputs.float())

