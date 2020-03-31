from keras.layers import Input, Dense, Flatten, Concatenate
from keras.models import Model
from keras import regularizers
import numpy as np
import pandas as pd
import keras
import pickle

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

seq_input = Input(shape = (21,5))
flat_seq = Flatten()(seq_input)
clip_input = Input(shape = (3, 6))
flat_clip = Flatten()(clip_input)
sig_input = Input(shape = (1, 21))
flat_sig = Flatten()(sig_input)
shape_input = Input(shape = (1, 244))
flat_shape = Flatten()(shape_input)

inputs = Concatenate()([flat_seq, flat_clip, flat_sig, flat_shape])
y = Dense(2000, activation='relu', W_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),
                                   b_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001))(inputs)
y = Dense(2000, activation='relu', W_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),
                                   b_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001))(y)
y = Dense(2000, activation='relu', W_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),
                                   b_regularizer = regularizers.l1_l2(l1=0.01,l2=0.01))(y)
out = Dense(1, activation = 'relu')(y)

model = Model(inputs = [seq_input,clip_input,sig_input,shape_input], outputs = out)
opt = keras.optimizers.adam(lr=0.0003)
model.compile(optimizer = opt, loss = 'mean_squared_error')
model.fit([gen, clip, signal, seq_shape], score, epochs = 15, batch_size = 16)
model.save_weights('DNN.h5')

