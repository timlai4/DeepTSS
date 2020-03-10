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
with open('all','rb') as f:
    test = pickle.load(f)
status = df.iloc[test].status
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

# Randomly shuffle the data
shuffle = np.random.rand(gen.shape[0]).argsort()
np.take(gen, shuffle, axis = 0, out = gen)
np.take(signal, shuffle, axis = 0, out = signal)
np.take(clip, shuffle, axis = 0, out = clip)
np.take(seq_shape, shuffle, axis = 0, out = seq_shape)
np.take(status, shuffle, axis = 0, out = status)

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
net = Net(input_size)
net = net.float()
# Weights with BCEWithLogitsLoss
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=1e-4)

print(device)
net.to(device)
inputs, status = inputs.to(device), status.to(device)
num_epochs = 20
bs = 16
for epoch in range(num_epochs):
    running_loss = 0.0
#    losses = []
    for i in range(0, inputs.size(0), bs):
        input_batch = inputs[i:i + bs, :]
        status_batch = status[i:i + bs, :]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(input_batch.float())
        loss = criterion(outputs, status_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
#        losses.append(loss.data.numpy())
        if i/bs % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
PATH = 'dnn.pth'
torch.save(net.state_dict(), PATH)

