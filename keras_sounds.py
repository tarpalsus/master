# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:39:39 2017

"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from guitar_pro import inverse_mapper
import pandas as pd
import numpy as np
import random
import os
import re

def prepare_model(maxlen, sound_idx):
    model = Sequential()
    model.add(LSTM(512, input_shape=(maxlen, len(sound_idx)), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, input_shape=(maxlen, len(sound_idx)),
                   return_sequences=True, dropout_U=0.5))
    model.add(Dropout(0.5))
    model.add(LSTM(512, input_shape=(maxlen, len(sound_idx)), dropout_U=0.5))
    model.add(Dense(len(sound_idx)))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.001)
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


def one_sound_encoding(sound_tuple):
    encoding = np.zeros(24 * 6)
    for sound in sound_tuple:
        string = inverse_mapper[sound[1]]
        encoding[(string-1) * 6 + int(sound[0])] = 1
    return encoding




def preprocess(path, maxlen, durations=False):
    df = pd.DataFrame()
    #for file in os.listdir(path): #TO IMPLEMENT
     #  df = pd.concat([df,pd.read_csv(os.path.join(path,file))])
    df = pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\black\black.csv")
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\ride\ride.csv")])
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\killem\killem.csv")])
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\master\master.csv")])
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\justice\justice.csv")])
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
    #distinct_sounds = []
    r = re.compile("([0-9]+)([a-zA-Z]+)*")
#    for i, sound in enumerate(sounds):
#        try:
#            sound_tuple = r.findall(sound)
#            distinct_sounds.append(one_sound_encoding(sound_tuple))
#        except Exception as e:
#            print(e)
#            pass
    sound_idx = dict((c, i) for i, c in enumerate(distinct_sounds))

    idx_sound = dict((i, c) for i, c in enumerate(distinct_sounds))

    for i in range(0, len(sounds) - maxlen, step):
        primary_sounds.append(sounds[i: i + maxlen])
        next_sounds.append(sounds[i + maxlen])
    #X = np.array(primary_sounds)
    #y = np.array(next_sounds)
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
              maxlen = 3,
              batch=128):
    model = prepare_model(maxlen, sound_idx)

    nb_epoch = 5
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
                      nb_epoch=10, verbose = 1)

    return model


def myGenerator(primary_sounds, distinct_sounds, next_sounds, sound_idx,batch,maxlen=20):
        #for i in (0, len(primary_sounds)-batch, batch):
        for i in (0,batch*2,batch):
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
    model.fit_generator(generator, samples_per_epoch=int(len(primary_sounds)), epochs=2)
    return model


def predict(soundList, maxlen, distinct_sounds, sound_idx, idx_sound,
            model, num_samples = 20):
    start_sequence = select_random_seed(soundList, maxlen)
    low = []
    high = []
    mid = []
    low = chain_predictions(model, start_sequence, 0.1, maxlen, num_samples,
                            distinct_sounds, sound_idx)
    mid = chain_predictions(model, start_sequence, 0.5, maxlen, num_samples,
                            distinct_sounds, sound_idx)
    high = chain_predictions(model, start_sequence, 1, maxlen, num_samples,
                            distinct_sounds, sound_idx)
    return low, mid, high


def chain_predictions(model, start_sequence, temperature, maxlen, num_samples,
                      distinct_sounds, sound_idx):
    output = []
    for i in range(num_samples):
        x = np.zeros((1, maxlen, len(distinct_sounds)))

        for i, sound in enumerate(start_sequence):
            x[0, i, sound_idx[sound]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_sound = idx_sound[next_index]
        output.append(next_sound)
        start_sequence = start_sequence[1:]
        start_sequence.append(next_sound)
    return output


MAX_LEN = 20
path = r"Metallica_csv"
sounds, primary_sounds, next_sounds, distinct_sounds, sound_idx, idx_sound = preprocess(path, MAX_LEN, durations=True)
trained_model = train_rnn(primary_sounds, next_sounds, distinct_sounds, sound_idx, MAX_LEN)
#trained_model = train_rnn_generator(primary_sounds, next_sounds,distinct_sounds,sound_idx)


low, mid, high = predict(sounds, MAX_LEN, distinct_sounds, sound_idx, idx_sound,
                         trained_model)