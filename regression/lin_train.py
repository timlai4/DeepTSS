import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

class TSSDataSet(Dataset):
    def __init__(self, data_dir, indices):
        '''
        Parameters
        ----------
        data_dir : Directory containing training data for classification
        '''
        allind, ind, softind = indices
        df = pd.read_csv(data_dir + 'raw_data.tsv',sep='\t')
        status = np.array([1 if score >= 10 else 0 for score in df['score']])
        df['status'] = status
        self.data = df.iloc[allind]
        
        gen = np.load(data_dir + 'genomic.npy')
        gen = np.reshape(gen[ind], (gen[ind].shape[0], gen[ind].shape[1], gen[ind].shape[2]))
        signal = np.load(data_dir + 'signal.npy')
        signal = np.reshape(signal[ind], (signal[ind].shape[0], signal[ind].shape[1], signal[ind].shape[2]))
        clip = np.load(data_dir + 'softclipped.npy')
        clip = np.reshape(clip[softind], (clip[softind].shape[0], clip[softind].shape[1], clip[softind].shape[2]))
        seq_shape = np.load(data_dir + 'shape.npy')
        seq_shape = np.reshape(seq_shape[ind], (seq_shape[ind].shape[0], seq_shape[ind].shape[1], seq_shape[ind].shape[2]))     
        self.inputs = [gen, signal, clip, seq_shape]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        status = np.array(self.data.iloc[idx]['status'])
        status = status.astype('float')
        gen, signal, clip, seq_shape = self.inputs
        gen = gen[idx]
        signal = signal[idx]
        clip = clip[idx]
        seq_shape = seq_shape[idx]
        sample = ([gen, signal, clip, seq_shape], status)
        return sample
    
class Linear(nn.Module):

    def __init__(self, input_size):
        super(Linear, self).__init__()        
        self.out = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.out(x)
        return x

# Load and process the data
data_dir = 'data/yeast_untreated/'
with open(data_dir + 'train-indices','rb') as f:
    allind, ind, softind = pickle.load(f)
ds = TSSDataSet(data_dir = data_dir, indices = [allind, ind, softind])
bs = 8 # batch size
dataloader = DataLoader(ds, batch_size = bs, shuffle = True)
batch = next(iter(dataloader))
gen, signal, clip, seq_shape = batch[0]

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
def train_model(model, criterion, optimizer, num_epochs = 25):
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        for inputs, status in dataloader:
            status = status.view(-1,1)
            gen, signal, clip, seq_shape = inputs
            gen = torch.flatten(gen, start_dim = 1)
            signal = torch.flatten(signal, start_dim = 1)
            clip = torch.flatten(clip, start_dim = 1)
            seq_shape = torch.flatten(seq_shape, start_dim = 1)
            inputs = torch.cat((gen, signal, clip, seq_shape), 1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, status)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2020 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
            i += 1
    #print(losses[-100:])
    print('Finished Training')
    PATH = model.__class__.__name__ + '.pth'
    torch.save(model.state_dict(), PATH)
    
    return model

lin = train_model(lin, criterion, optimizer, num_epochs)