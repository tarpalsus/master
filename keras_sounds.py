# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:39:39 2017

"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
import pandas as pd
import numpy as np
import random
import os


def prepare_model(maxlen, sound_idx):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(sound_idx))))
    model.add(Dense(len(sound_idx)))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def select_random_seed(soundList, maxlen):
    start_index = random.randint(0, len(soundList) - maxlen - 1)
    seed = soundList[start_index: start_index + maxlen]
    return seed


def preprocess(path, maxlen, durations=False):
    df = pd.DataFrame()
    for file in os.listdir(path): #TO IMPLEMENT
       df = pd.concat([df,pd.read_csv(os.path.join(path,file))])
#    df = pd.read_csv(r"C:\Users\Maciek\Downloads\black\black.csv")
#    df = pd.concat([df, pd.read_csv(r"C:\Users\Maciek\Downloads\ride\ride.csv")])
#    df = pd.concat([df, pd.read_csv(r"C:\Users\Maciek\Downloads\killem\killem.csv")])
#    df = pd.concat([df, pd.read_csv(r"C:\Users\Maciek\Downloads\master\master.csv")])
#    df = pd.concat([df, pd.read_csv(r"C:\Users\Maciek\Downloads\justice\justice.csv")])
    df['Duration'] = df['Duration'].apply(int)
    df['Duration'] = df['Duration'].apply(str)
    df.dropna(inplace=True)
    if durations:
        sounds = df['Value+String'].str.cat(df['Duration'])
    else:
        sounds = df['Value+String']
    sounds.apply(str)
    sounds = list(sounds)
    distinct_sounds = sorted(set(sounds))
    step = 1
    primary_sounds = []
    next_sounds = []
    sound_idx = dict((c, i) for i, c in enumerate(distinct_sounds))

    idx_sound = dict((i, c) for i, c in enumerate(distinct_sounds))

    for i in range(0, len(sounds) - maxlen, step):
        primary_sounds.append(sounds[i: i + maxlen])
        next_sounds.append(sounds[i + maxlen])
#    X = np.zeros((len(primary_sounds), maxlen, len(distinct_sounds)),
#                 dtype=np.bool)
#
#    y = np.zeros((len(primary_sounds), len(distinct_sounds)), dtype=np.bool)
#
#    for i, chunk in enumerate(primary_sounds):
#        for t, sound in enumerate(chunk):
#            X[i, t, sound_idx[sound]]
#        y[i, sound_idx[next_sounds[i]]] = 1
#    print('ok')

    return sounds, primary_sounds, next_sounds,  distinct_sounds, sound_idx, idx_sound


def train_rnn(primary_sounds, next_sounds,distinct_sounds,sound_idx,
              maxlen = 20,
              batch=128):
    model = prepare_model(maxlen, sound_idx)

    nb_epoch = 20
    for e in range(nb_epoch):
        for i in range(0, len(primary_sounds)-batch, batch):
            X = np.zeros((batch, maxlen, len(distinct_sounds)),
                             dtype=np.bool)
            y = np.zeros((batch, len(distinct_sounds)), dtype=np.bool)
            for i, chunk in enumerate(primary_sounds[i:i+batch]):
                for t, sound in enumerate(chunk):
                    X[i, t, sound_idx[sound]] = 1
                y[i, sound_idx[next_sounds[i]]] = 1
            model.fit(X, y,
                      batch_size=batch,
                      nb_epoch=1, verbose = 1)

    return model


def myGenerator(primary_sounds, distinct_sounds, next_sounds, sound_idx,batch,maxlen=20):
    while 1:
        for i in (0, len(primary_sounds)-batch, batch):
            #if i%125==0:
               # print ("i = " + str(i))
            print(i)
            X = np.zeros((batch, maxlen, len(distinct_sounds)),
                             dtype=np.bool)
            y = np.zeros((batch, len(distinct_sounds)), dtype=np.bool)
            for i, chunk in enumerate(primary_sounds[i:i+batch]):
                for t, sound in enumerate(chunk):
                    X[i, t, sound_idx[sound]] = 1
                y[i, sound_idx[next_sounds[i]]] = 1

            yield (X, y)


def train_rnn_generator(primary_sounds, next_sounds,distinct_sounds,sound_idx,
              maxlen = 20,
              batch=128):

    model = prepare_model(maxlen, sound_idx)
    generator = myGenerator(primary_sounds, distinct_sounds, next_sounds, sound_idx, batch)
    model.fit_generator(generator, samples_per_epoch=int(len(primary_sounds)), nb_epoch=2)
    return model


def predict(soundList, maxlen, distinct_sounds, sound_idx, idx_sound,
            model, num_samples = 20):
    sentence = select_random_seed(soundList, maxlen)
    low = []
    high = []
    mid = []
    for i in range(num_samples):
        x = np.zeros((1, maxlen, len(distinct_sounds)))

        for i, sound in enumerate(sentence):

            x[0, i, sound_idx[sound]] = 1.



        preds = model.predict(x, verbose=0)[0]
        low.append(idx_sound[sample(preds, 0.1)])
        mid.append(idx_sound[sample(preds, 0.5)])
        high.append(idx_sound[sample(preds, 1)])
        next_index = sample(preds, 0.7)
        next_sound = idx_sound[next_index]
        sentence = sentence[1:]
        sentence.append(next_sound)
    return low, mid, high

MAX_LEN = 20
path = r"Metallica_csv"
sounds, primary_sounds, next_sounds, distinct_sounds, sound_idx, idx_sound = preprocess(path, MAX_LEN, durations=True)
#trained_model = train_rnn(primary_sounds, next_sounds, distinct_sounds, sound_idx)
trained_model = train_rnn_generator(primary_sounds, next_sounds,distinct_sounds,sound_idx)


low, mid, high = predict(sounds, MAX_LEN, distinct_sounds, sound_idx, idx_sound,
                         trained_model)