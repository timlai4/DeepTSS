import torch
import torch.nn as nn
import torch.optim as optim
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
data_dir = 'data/yeast_untreated/'
df = pd.read_csv(data_dir + 'raw_data.tsv',sep='\t')
with open(data_dir + 'train-indices','rb') as f:
    all, ind, softind = pickle.load(f)
df = df.iloc[all]
score = np.array(np.log(df['score']))

gen = np.load(data_dir + 'genomic.npy')
gen = np.reshape(gen[ind], (gen[ind].shape[0], gen[ind].shape[1], gen[ind].shape[2]))
signal = np.load(data_dir + 'signal.npy')
signal = np.reshape(signal[ind], (signal[ind].shape[0], signal[ind].shape[1], signal[ind].shape[2]))
clip = np.load(data_dir + 'softclipped.npy')
clip = np.reshape(clip[softind], (clip[softind].shape[0], clip[softind].shape[1], clip[softind].shape[2]))
seq_shape = np.load(data_dir + 'shape.npy')
seq_shape = np.reshape(seq_shape[ind], (seq_shape[ind].shape[0], seq_shape[ind].shape[1], seq_shape[ind].shape[2]))

# Randomly shuffle the data
shuffle = np.random.rand(gen.shape[0]).argsort()
np.take(gen, shuffle, axis = 0, out = gen)
np.take(signal, shuffle, axis = 0, out = signal)
np.take(clip, shuffle, axis = 0, out = clip)
np.take(seq_shape, shuffle, axis = 0, out = seq_shape)
np.take(score, shuffle, axis = 0, out = score)

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
lin = lin.float()
# Weights with BCEWithLogitsLoss
criterion = nn.MSELoss()
optimizer = optim.Adam(lin.parameters(), lr=0.0003)

num_epochs = 60
bs = 8
for epoch in range(num_epochs):
    running_loss = 0.0
#    losses = []
    for i in range(0, inputs.size(0), bs): 
        input_batch = inputs[i:i + bs, :] 
        score_batch = score[i:i + bs, :]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = lin(input_batch.float())
        loss = criterion(outputs, score_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i/bs % 2000 == 1999:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
PATH = 'lin.pth'
torch.save(lin.state_dict(), PATH)


