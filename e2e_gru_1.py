import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import GRU, Bidirectional, Dense, Dropout
from keras.layers import Input, concatenate
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def net():
    input_audio = Input(shape=(400, 36))
    input_eyes = Input(shape=(200, 6))
    input_face = Input(shape=(200, 100))
    input_kinect = Input(shape=(60, 27))

    x_audio = Bidirectional(GRU(30, return_sequences=True), input_shape=(400, 36))(input_audio)
    x_audio = Dropout(0.1)(x_audio)
    embd_audio = Bidirectional(GRU(15))(x_audio)

    x_eyes = Bidirectional(GRU(6, return_sequences=True, input_shape=(200, 6)))(input_eyes)
    x_eyes = Dropout(0.1)(x_eyes)
    embd_eyes = Bidirectional(GRU(6))(x_eyes)

    x_face = Bidirectional(GRU(50, return_sequences=True), input_shape=(200, 100))(input_face)
    x_face = Dropout(0.1)(x_face)
    emdb_face = Bidirectional(GRU(20))(x_face)

    x_kinect = Bidirectional(GRU(20, return_sequences=True), input_shape=(60, 27))(input_kinect)
    x_kinect = Dropout(0.1)(x_kinect)
    embd_kinect = Bidirectional(GRU(10))(x_kinect)

    fc = concatenate([embd_audio, embd_eyes, emdb_face, embd_kinect])
    fc = Dropout(0.5)(fc)
    fc = Dense(50, activation='relu')(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(6, activation='softmax')(fc)

    model = Model((input_audio, input_eyes, input_face, input_kinect), fc)

    return model

model = net()
model.summary()
opt = Adam(lr=0.001)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

mins_audio, maxs_audio = np.load('audio_mins.npy'), np.load('audio_maxs.npy')
mins_kinect, maxs_kinect = np.load('kinect_mins.npy'), np.load('kinect_maxs.npy')

def norm(arr, mins, maxs):
    arr = arr.astype(float)
    for i in range(arr.shape[-1]):
        arr[:, :, i] -= mins[i]
        arr[:, :, i] /= maxs[i]
    return arr

X_train_audio = norm(np.load('X_train_audio.npy'), mins_audio, maxs_audio)
X_val_audio = norm(np.load('X_val_audio.npy'), mins_audio, maxs_audio)

X_train_eyes = np.load('X_train_eyes.npy')
X_val_eyes = np.load('X_val_eyes.npy')

X_train_face = np.load('X_train_face.npy')
X_val_face = np.load('X_val_face.npy')

X_train_kinect = norm(np.load('X_train_kinect.npy'), mins_kinect, maxs_kinect)
X_val_kinect = norm(np.load('X_val_kinect.npy'), mins_kinect, maxs_kinect)

y_train = to_categorical(np.load('y_train.npy'))
y_val = to_categorical(np.load('y_val.npy'))

mc = ModelCheckpoint('model_e2e_1.hf5', verbose=1, save_best_only=True)

model.fit([X_train_audio, X_train_eyes, X_train_face, X_train_kinect], y_train, 
    epochs=20, batch_size=100,
    validation_data=([X_val_audio, X_val_eyes, X_val_face, X_val_kinect], y_val),
    callbacks=[mc])
