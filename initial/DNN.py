from keras.layers import Input, Dense, Flatten, Concatenate
from keras.models import Model
from keras import regularizers
import numpy as np
import pandas as pd
import keras
import pickle

df = pd.read_csv('sequence_data.tsv',sep='\t')
with open('all','rb') as f:
    test = pickle.load(f)

status = df.iloc[test].status
status = np.array(status)

pos = 6931
neg = 101232
total = pos + neg

weight_0 = (1 / neg)*(total) / 2.0
weight_1 = (1 / pos)*(total) / 2.0

weights = {0: weight_0, 1: weight_1}
keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall')]

gen = np.load('genomic.npy')
gen = np.reshape(gen[test], (gen[test].shape[0], gen[test].shape[1], gen[test].shape[2]))
signal = np.load('signal.npy')
signal = np.reshape(signal[test], (signal[test].shape[0], signal[test].shape[1], signal[test].shape[2]))
clip = np.load('softclipped.npy')
clip = np.reshape(clip[test], (clip[test].shape[0], clip[test].shape[1], clip[test].shape[2]))
seq_shape = np.load('shape.npy')
seq_shape = np.reshape(seq_shape[test], (seq_shape[test].shape[0], seq_shape.shape[1], seq_shape.shape[2]))

seq_input = Input(shape = (21,5))
flat_seq = Flatten()(seq_input)
clip_input = Input(shape = (clip.shape[1], clip.shape[2]))
flat_clip = Flatten()(clip_input)
sig_input = Input(shape = (signal.shape[1], signal.shape[2]))
flat_sig = Flatten()(sig_input)
shape_input = Input(shape = (seq_shape.shape[1], seq_shape.shape[2]))
flat_shape = Flatten()(shape_input)

inputs = Concatenate()([flat_seq, flat_clip, flat_sig, flat_shape])
y = Dense(2000, activation='relu', W_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),
                                   b_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001))(inputs)
y = Dense(2000, activation='relu', W_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),
                                   b_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001))(y)
y = Dense(2000, activation='relu', W_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),
                                   b_regularizer = regularizers.l1_l2(l1=0.01,l2=0.01))(y)
out = Dense(1, activation = 'sigmoid')(y)

model = Model(inputs = [seq_input,clip_input,sig_input,shape_input], outputs = out)
opt = keras.optimizers.adam(lr=0.0003)
model.compile(optimizer = opt, loss = 'binary_crossentropy',
                                  metrics = metrics)
# Do the training without changing loss function, then save weights:
model.fit([gen, clip, signal, seq_shape], status, epochs = 10, batch_size = 32, validation_split = 0.1)
model.save_weights('DNN_unbalance.h5')
# Do the training balancing the loss function, then save weights:
model.fit([gen, clip, signal, seq_shape], status, epochs = 10, batch_size = 32, validation_split = 0.1, class_weight = weights)
model.save_weights('DNN_balance.h5')

