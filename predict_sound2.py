# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:49:24 2017


"""

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


import os
import random

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import grid_search
from guitar_pro import inverse_mapper, string_mapper, rnn_to_guitar_pro
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import gensim
import time
import re

#from legacy import get_data, plot_with_labels

import tflearn
from tflearn.data_utils import *
from scipy.sparse import csc_matrix

duration_encoding_mapper = {
        1: 0,
        2: 1,
        4: 2,
        8: 3,
        16: 4,
        32: 5,
        64: 6
        }

inverse_duration_mapper = {v: k for k, v in duration_encoding_mapper.items()}


def test_gensim():
    """Unused test of gensim library"""
    soundList = get_data('black')
    soundList.extend(get_data('killem'))
    soundList.extend(get_data('ride'))
    soundList.extend(get_data('master'))
    soundList.extend(get_data('justice'))
    sentences = []
    for tab in soundList:
        sentences.append([tab[i:i + 15] for i in range(0, len(tab), 15)])
    sentences2 = []
    for sentence in sentences:
        for x in sentence:
            sentences2.append(x)
    sentences2 = [' '.join(sentence).lower() for sentence in sentences2]
    sentencesFinal = [sentence.split() for sentence in sentences2]
    model = gensim.models.Word2Vec(sentencesFinal, size=300, window=10, sg=0)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    #pca = PCA(n_components =2)
    #T = pca.fit_transform(model.wv.syn0)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    T = tsne.fit_transform(model.wv.syn0)
    plot_with_labels(T, model.wv.index2word, filename='pca.png')
    return model, sentencesFinal, soundList


def one_sound_encoding(sound_tuple):
    encoding = np.zeros(24 * 6)
    for sound in sound_tuple:
        string = inverse_mapper[sound[1]]
        encoding[(string-1) * 6 + int(sound[0])] = 1
    return encoding


def one_sound_encoding(sound,r, durations):
    string_number = len(inverse_mapper)
    duration_number = len(duration_encoding_mapper)
    sound_number = 26
    if durations:
        duration = duration_encoding_mapper[int(sound[1])]
        sound = sound[0]
        encoding = np.zeros(sound_number * string_number * duration_number)
    else:
        encoding = np.zeros(sound_number * string_number)

    for string in inverse_mapper.keys():
        if string in sound:
            sound = sound
        else:
            sound = sound + '25' + str(string)
    sound_tuple = r.findall(sound)
    for sound in sound_tuple:
        string = inverse_mapper[sound[1]]
        if not durations:
            encoding[(string-1) * sound_number + int(sound[0])] = 1
        else:
            encoding[((string-1) * duration_number + duration) * sound_number + int(sound[0])] = 1
    return encoding


def prepare_data(path, maxlen,durations=False):
    """Get useful data from csv files, generate """
    df = pd.DataFrame()
    for file in os.listdir(path): #TO IMPLEMENT
        df = pd.concat([df, pd.read_csv(os.path.join(path, file))])
    df = pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\black\black.csv")
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\ride\ride.csv")])
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\killem\killem.csv")])
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\master\master.csv")])
    df = pd.concat([df, pd.read_csv(r"C:\Users\user\Desktop\Sound_generator\justice\justice.csv")])

    df['Duration'] = df['Duration'].apply(int)
    df['Duration'] = df['Duration'].apply(str)
    df.dropna(inplace=True)
    if durations:
        sounds = list(zip(df['Value+String'], df['Duration']))
    else:
        sounds = df['Value+String']
    #sounds.apply(str)
    sounds = list(sounds)
    distinct_sounds = sorted(set(sounds))
    print(len(distinct_sounds))
    transformed_sounds = []
    step = 1
    
    r = re.compile("([0-9]+)([a-zA-Z]+)*")
    sound_idx = dict((c, i) for i, c in enumerate(distinct_sounds))
    idx_sound = dict((i, c) for i, c in enumerate(distinct_sounds))
    return sound_idx, idx_sound, sounds, distinct_sounds

def preprocess(sounds, distinct_sounds, maxlen, sound_idx):
    step = 1
    primary_sounds = []
    next_sounds = []
    for i in range(0, len(sounds) - maxlen, step):
        primary_sounds.append(sounds[i: i + maxlen])
        next_sounds.append(sounds[i + maxlen])



    X = np.zeros((len(primary_sounds), maxlen, len(distinct_sounds)),
                 dtype=np.bool)

    y = np.zeros((len(primary_sounds), len(distinct_sounds)), dtype=np.bool)

    for i, chunk in enumerate(primary_sounds):

        for t, sound in enumerate(chunk):
            X[i, t, sound_idx[sound]] = 1
        y[i, sound_idx[next_sounds[i]]] = 1
    X, y = shuffle(X,y)
    return X, y


def select_random_seed(soundList, maxlen):
    start_index = random.randint(0, len(soundList) - maxlen - 1)
    seed = soundList[start_index: start_index + 1]
    return seed


def build_and_run_model(soundList, maxlen, sound_idx,
                        distinct_sounds,
                        seq_len=50, out_file='model.tflearn',  iters=20,
                        durations=False):
    """ Build and create sequences with different temperatures (measure of "innovation")"""
    tf.reset_default_graph()
    with tf.Graph().as_default():
        net = tflearn.input_data([None, maxlen, len(sound_idx)])
        net = tflearn.lstm(net, 512, return_seq=True)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.lstm(net, 512, return_seq=True)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.lstm(net, 512)
        net = tflearn.fully_connected(net, len(sound_idx),
                                      activation='softmax')
        net = tflearn.regression(net, optimizer='adam',
                                 loss='categorical_crossentropy',
                                 learning_rate=0.001)
        model = tflearn.SequenceGenerator(net, dictionary=sound_idx,
                                          seq_maxlen=maxlen,
                                          clip_gradients=5.0, tensorboard_verbose=3)
#    X = np.zeros((len(primary_sounds), maxlen, len(distinct_sounds)),
#                             dtype=np.bool)
#    y = np.zeros((len(primary_sounds), len(distinct_sounds)), dtype=np.bool)
#    for i, chunk in enumerate(primary_sounds):
#
#        for t, sound in enumerate(chunk):
#            X[i, t, sound_idx[sound]] = 1
#        y[i, sound_idx[next_sounds[i]]] = 1
#    for i in range(0, len(primary_sounds)-batch, batch):
#            X = np.zeros((batch, maxlen, len(distinct_sounds)),

#                             dtype=np.bool)
#            y = np.zeros((batch, len(distinct_sounds)), dtype=np.bool)
#            for i, chunk in enumerate(primary_sounds[i:i+batch]):
#                for t, sound in enumerate(chunk):
#                    X[i, t, sound_idx[sound]] = 1
#                y[i, sound_idx[next_sounds[i]]] = 1
    load = False
    big_batch = 2* 42870
    for i in range(0, len(soundList), big_batch):
        X, y = preprocess(soundList[i:i+big_batch], distinct_sounds, maxlen, sound_idx)
        if not load:
            model.fit(X, y, validation_set=0.1, batch_size=128, n_epoch=1)
            #model.save(out_file)
        else:
            pass
            #model = model.load(out_file)
    print('Trained')
    temperatures_low = []
    temperatures_high = []
    temperatures_mid = []
    for i in range(iters):
        seed = select_random_seed(soundList, maxlen)
        temperatures_low.append(model.generate(seq_len, temperature=0.1,
                                               seq_seed=seed))
        temperatures_mid.append(model.generate(seq_len, temperature=0.5,
                                               seq_seed=seed))
        temperatures_high.append(model.generate(seq_len, temperature=1,
                                                seq_seed=seed))
#        temperatures_high.append(model.generate(seq_len, temperature=1,
#                                                seq_seed=seed))
        print(i)
    print('Sequences generated')
    return temperatures_high, temperatures_low, temperatures_mid


def infer_sound(prediction, durations):
    result = np.zeros((0, 1))
    untransformed_sound = []
    string_split = np.split(prediction, 6)
    for i, string_sound in enumerate(string_split):
        if durations:
            result_string = np.zeros((0, 1))
            durations_split = np.split(string_sound, 7)
            for j, duration_sound in enumerate(durations_split):
                result_duration = np.zeros_like(durations_split)
                index = np.argmax(duration_sound)
                result_duration[index] = 1
                duration = inverse_duration_mapper[j]
                result_string = np.concatenate((result_string, result_duration))
                string = string_mapper[i + 1]
                untransformed_sound.append((str(index) + str(string), duration))
            result = np.concatenate((result, result_string))
        else:
            result_string = np.zeros_like(string_sound)
            index = np.argmax(string_sound)
            result_string[index] = 1
            string = string_mapper[i + 1]
            untransformed_sound.append(str(index) + str(string))
            result = np.concatenate((result, result_string))
    return untransformed_sound


if __name__ == '__main__':

    MAX_LEN = 20
    path = r"Metallica_csv"
    #Process all files in folder, currently beyond scope of my
    #machines computational abilities. For smaller dataset, pick csv file created by
    #process albums function

    #primary_sounds, next_sounds, sound_idx = prepare_data(path, MAX_LEN,durations=True)
    durations = True
    sound_ids, idx_sound, soundList, distinct_sounds = prepare_data(path, MAX_LEN,durations=durations)
    X, y = preprocess(soundList, distinct_sounds, MAX_LEN, sound_ids)
    print(X.shape)
    print(y.shape)


#
    high, low, mid = build_and_run_model(soundList, MAX_LEN, sound_ids, distinct_sounds,
                                         seq_len=200,
                                         out_file='10ep_dropout_len20.tflearn',
                                         durations = durations)
    rnn_to_guitar_pro(high,'test_new',durations,False)
    with open('high_10ep_drop_len7.pkl', 'wb') as f:
        pickle.dump(high, f)

    with open('mid_10ep_drop_len7.pkl', 'wb') as f:
        pickle.dump(mid, f)

    with open('low_10ep_drop_len7.pkl', 'wb') as f:
        pickle.dump(low, f)
    