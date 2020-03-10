from keras.layers import Input, Dense, Flatten, Concatenate
from keras.models import Model
from keras import regularizers
from keras.utils import CustomObjectScope
import numpy as np
import pandas as pd
import keras
import pickle

df = pd.read_csv('sequence_data.tsv',sep='\t')
test = df['status'] < 2
status = df[test].status
status = np.array(status)

metrics = [keras.metrics.AUC(name='auc'),
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
clip_input = Input(shape = (3, 6))
flat_clip = Flatten()(clip_input)
sig_input = Input(shape = (1, 21))
flat_sig = Flatten()(sig_input)
shape_input = Input(shape = (1, 244))
flat_shape = Flatten()(shape_input)

inputs = Concatenate()([flat_seq, flat_clip, flat_sig, flat_shape])
out = Dense(1, activation = 'sigmoid')(inputs)

model1 = Model(inputs = [seq_input,clip_input,sig_input,shape_input], outputs = out)
model2 = Model(inputs = [seq_input,clip_input,sig_input,shape_input], outputs = out)
with CustomObjectScope({'metrics': metrics}):
    model1.load_weights('logit_balance.h5')
    model2.load_weights('logit_unbalance.h5')
opt = keras.optimizers.adam(lr=0.0003)
model1.compile(optimizer = opt, loss = 'binary_crossentropy',
                                  metrics = ['accuracy'])
model2.compile(optimizer = opt, loss = 'binary_crossentropy',
                                  metrics = ['accuracy'])
print(model1.summary())


balance_perf = model1.evaluate([gen, clip, signal, seq_shape], status, use_multiprocessing = True)

unbalance_perf = model2.evaluate([gen, clip, signal, seq_shape], status, use_multiprocessing = True)
balance_pred = model1.predict([gen, clip, signal, seq_shape], use_multiprocessing = True)
unbalance_pred = model2.predict([gen, clip, signal, seq_shape], use_multiprocessing = True)

with open('logbalanced_performance','wb') as f:
    pickle.dump(balance_perf,f)
with open('logunbalanced_performance','wb') as f:
    pickle.dump(unbalance_perf,f)
with open('logbalance_pred','wb') as f:
    pickle.dump(balance_pred,f)
with open('logunbalance_pred','wb') as f:
    pickle.dump(unbalance_pred,f)


#model.fit([gen, clip, signal, seq_shape], status, epochs = 10, batch_size = 32, validation_split = 0.1, class_weight = weights)
#model.save_weights('DNN_balance.h5')

