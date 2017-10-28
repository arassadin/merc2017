import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.layers import Input, concatenate, average
from keras.utils import to_categorical
from keras.optimizers import Adam

def net():
    input_audio = Input(shape=(400, 36))
    input_eyes = Input(shape=(200, 6))
    input_face = Input(shape=(200, 100))
    input_kinect = Input(shape=(60, 27))

    x_audio = Bidirectional(LSTM(100, return_sequences=True), input_shape=(400, 36))(input_audio)
    embd_audio = Bidirectional(LSTM(100))(x_audio)

    x_eyes = Bidirectional(LSTM(100, return_sequences=True, input_shape=(200, 6)))(input_eyes)
    embd_eyes = Bidirectional(LSTM(100))(x_eyes)

    x_face = Bidirectional(LSTM(100, return_sequences=True), input_shape=(200, 100))(input_face)
    emdb_face = Bidirectional(LSTM(100))(x_face)

    x_kinect = Bidirectional(LSTM(100, return_sequences=True), input_shape=(60, 27))(input_kinect)
    embd_kinect = Bidirectional(LSTM(100))(x_kinect)

    fc = average([embd_audio, embd_eyes, emdb_face, embd_kinect])
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

model.fit([X_train_audio, X_train_eyes, X_train_face, X_train_kinect], y_train, 
    epochs=30, batch_size=10,
    validation_data=([X_val_audio, X_val_eyes, X_val_face, X_val_kinect], y_val))
