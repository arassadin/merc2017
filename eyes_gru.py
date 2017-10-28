import os
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def net():
    model = Sequential()
    model.add(Bidirectional(GRU(6, return_sequences=True), input_shape=(200, 6)))
    model.add(Dropout(0.6))
    model.add(Bidirectional(GRU(6)))
    model.add(Dropout(0.6))
    model.add(Dense(6, activation='softmax'))
    return model

model = net()
model.summary()
opt = Adam(lr=0.001)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

X_train = np.load('X_train_eyes.npy')
y_train = to_categorical(np.load('y_train.npy'))
X_val = np.load('X_val_eyes.npy')
y_val = to_categorical(np.load('y_val.npy'))

print X_train.shape, y_train.shape
print X_val.shape, y_val.shape

print X_train.min(), X_train.max()
print X_val.min(), X_val.max()

mc = ModelCheckpoint('model_eyes.h5', verbose=1, save_best_only=True)

model.fit(X_train, y_train, epochs=20, batch_size=100,
    validation_data=(X_val, y_val),
    callbacks=[mc])
