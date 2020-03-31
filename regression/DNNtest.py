from keras.layers import Input, Dense, Flatten, Concatenate
from keras.models import Model
from keras import regularizers
import numpy as np
import pandas as pd
import keras
import pickle

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
#model.fit([gen, clip, signal, seq_shape], score, epochs = 15, batch_size = 16)
model.load_weights('DNN.h5')
print(model.evaluate([gen,clip,signal,seq_shape], score, batch_size = 256))
y = model.predict([gen,clip,signal,seq_shape], batch_size = 256)
df['DNN_Pred'] = y
df.to_csv('pred.tsv', sep = '\t', index = False)
