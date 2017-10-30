import os
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

def net():
    model = Sequential()
    model.add(Bidirectional(GRU(30, return_sequences=True), input_shape=(400, 36)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(15)))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    return model

model = net()
model.summary()
opt = Adam(lr=0.001)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

mins, maxs = np.load('data/audio_mins.npy'), np.load('data/audio_maxs.npy')
def norm(arr):
    arr = arr.astype(float)
    for i in range(arr.shape[-1]):
        arr[:, :, i] -= mins[i]
        arr[:, :, i] /= maxs[i]
    return arr

X_train = norm(np.concatenate([np.load('data/X_train_audio.npy'), np.load('data/X_val_audio.npy')]))
y_train = to_categorical(np.concatenate([np.load('data/y_train.npy'), np.load('data/y_val.npy')]))

print X_train.shape, y_train.shape
print X_train.min(), X_train.max()

mc = ModelCheckpoint('models/model_audio_phase2.h5', verbose=1, save_best_only=True, monitor='loss')
tb = TensorBoard(log_dir='tflogs')

model.fit(X_train, y_train, epochs=30, batch_size=100, callbacks=[mc, tb])
