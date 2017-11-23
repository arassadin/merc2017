import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.layers import Input, concatenate, average
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

def net():
    input_audio = Input(shape=(400, 36))
    input_eyes = Input(shape=(200, 6))
    input_face = Input(shape=(200, 100))
    input_kinect = Input(shape=(60, 27))

    x_audio = Bidirectional(LSTM(100, return_sequences=True), input_shape=(400, 36))(input_audio)
    x_audio = Dropout(0.7)(x_audio)
    embd_audio = Bidirectional(LSTM(100))(x_audio)

    x_eyes = Bidirectional(LSTM(100, return_sequences=True, input_shape=(200, 6)))(input_eyes)
    x_eyes = Dropout(0.7)(x_eyes)
    embd_eyes = Bidirectional(LSTM(100))(x_eyes)

    x_face = Bidirectional(LSTM(100, return_sequences=True), input_shape=(200, 100))(input_face)
    x_face = Dropout(0.7)(x_face)
    emdb_face = Bidirectional(LSTM(100))(x_face)

    x_kinect = Bidirectional(LSTM(100, return_sequences=True), input_shape=(60, 27))(input_kinect)
    x_kinect = Dropout(0.7)(x_kinect)
    embd_kinect = Bidirectional(LSTM(100))(x_kinect)

    fc = average([embd_audio, embd_eyes, emdb_face, embd_kinect])
    fc = Dense(6, activation='softmax')(fc)

    model = Model((input_audio, input_eyes, input_face, input_kinect), fc)

    return model

model = net()
model.summary()
opt = Adam(lr=0.001)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

mins_audio, maxs_audio = np.load('data/audio_mins.npy'), np.load('data/audio_maxs.npy')
mins_kinect, maxs_kinect = np.load('data/kinect_mins.npy'), np.load('data/kinect_maxs.npy')

def norm(arr, mins, maxs):
    arr = arr.astype(float)
    for i in range(arr.shape[-1]):
        arr[:, :, i] -= mins[i]
        arr[:, :, i] /= maxs[i]
    return arr

X_audio = norm(np.concatenate([np.load('data/X_train_audio.npy'), np.load('data/X_val_audio.npy')]), mins_audio, maxs_audio)
X_eyes = np.concatenate([np.load('data/X_train_eyes.npy'), np.load('data/X_val_eyes.npy')])
X_face = np.concatenate([np.load('data/X_train_face.npy'), np.load('data/X_val_face.npy')])
X_kinect = norm(np.concatenate([np.load('data/X_train_kinect.npy'), np.load('data/X_val_kinect.npy')]), mins_kinect, maxs_kinect)

y = to_categorical(np.concatenate([np.load('data/y_train.npy'), np.load('data/y_val.npy')]))

mc = ModelCheckpoint('models/model_e2e_phase2.hf5', monitor='loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='tflogs')

model.fit([X_audio, X_eyes, X_face, X_kinect], y,
    epochs=30, batch_size=100, callbacks=[mc, tb])
