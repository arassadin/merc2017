import os
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam

def net():
    model = Sequential()
    model.add(Bidirectional(GRU(30, return_sequences=True), input_shape=(400, 36)))
    model.add(Bidirectional(GRU(15)))
    model.add(Dense(6, activation='softmax'))
    return model

model = net()
model.summary()
opt = Adam(lr=0.001)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

mins, maxs = np.load('audio_mins.npy'), np.load('audio_maxs.npy')
def norm(arr):
    arr = arr.astype(float)
    for i in range(arr.shape[-1]):
        arr[:, :, i] -= mins[i]
        arr[:, :, i] /= maxs[i]
    return arr

X_train = norm(np.load('X_train_audio.npy'))
y_train = to_categorical(np.load('y_train.npy'))
X_val = norm(np.load('X_val_audio.npy'))
y_val = to_categorical(np.load('y_val.npy'))

print X_train.shape, y_train.shape
print X_val.shape, y_val.shape

print X_train.min(), X_train.max()
print X_val.min(), X_val.max()

model.fit(X_train, y_train, epochs=10, batch_size=100,
    validation_data=(X_val, y_val))
