# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:49:24 2017


"""

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import dask.array as da


import os
import random

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import grid_search
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import gensim
import h5py
import time

#from legacy import get_data, plot_with_labels

import tflearn
from tflearn.data_utils import *
from scipy.sparse import csc_matrix


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





def prepare_data(path, maxlen,durations=False):
    """Get useful data from csv files, generate """
    df = pd.DataFrame()
    for file in os.listdir(path): #TO IMPLEMENT
       df = pd.concat([df,pd.read_csv(os.path.join(path,file))])
    #df = pd.read_csv(r"C:\Users\Maciek\Downloads\black\black.csv")
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

    X = np.zeros((len(primary_sounds), maxlen, len(distinct_sounds)),
                 dtype=np.bool)

    y = np.zeros((len(primary_sounds), len(distinct_sounds)), dtype=np.bool)

    for i, chunk in enumerate(primary_sounds):

        for t, sound in enumerate(chunk):
            X[i, t, sound_idx[sound]] = 1
        y[i, sound_idx[next_sounds[i]]] = 1

    return sounds, sound_idx, distinct_sounds, primary_sounds, next_sounds,f1, X, y


def select_random_seed(soundList, maxlen):
    start_index = random.randint(0, len(soundList) - maxlen - 1)
    seed = soundList[start_index: start_index + maxlen]
    return seed


def build_and_run_model(X,y,soundList, maxlen, sound_idx,
                        primary_sounds, next_sounds, distinct_sounds,
                        seq_len=50, out_file='model.tflearn',  iters=20):
""" Build and create sequences with different temperatures (measure of "innovation")"""
    with tf.Graph().as_default():
        net = tflearn.input_data([None, maxlen, len(sound_idx)],
                                  data_preprocessing=None,
                                  data_augmentation=None)
        #net = tflearn.lstm(net, 512, return_seq=True)
        #net = tflearn.dropout(net, 0.5)
        #net = tflearn.lstm(net, 512, return_seq=True)
        #net = tflearn.dropout(net, 0.5)
        net = tflearn.lstm(net, 512)
        net = tflearn.fully_connected(net, len(sound_idx),
                                      activation='softmax')
        net = tflearn.regression(net, optimizer='adam',
                                 loss='categorical_crossentropy',
                                 learning_rate=0.001)
        model = tflearn.SequenceGenerator(net, dictionary=sound_idx,
                                          seq_maxlen=maxlen,
                                          clip_gradients=5.0)
    batch = 128
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
    model.fit(X, y, validation_set=0.1, batch_size=128, n_epoch=1,
                      run_id='Metallica_drop_20 ')
    model.save(out_file)
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
        print(i)
    print('Sequences generated')
    return temperatures_high, temperatures_low, temperatures_mid


if __name__ == '__main__':
    MAX_LEN = 20
    path = r"Metallica_csv"
    #Process all files in folder, currently beyond scope of my
    #machines computational abilities. For smaller dataset, pick csv file created by
    #process albums function

    #primary_sounds, next_sounds, sound_idx = prepare_data(path, MAX_LEN,durations=True)

    soundList, sound_ids, distinct_sounds, primary_sounds, next_sounds,f1, X,y = prepare_data(path, MAX_LEN,durations=True)
    print(X.shape)
    print(y.shape)


#
    high, low, mid = build_and_run_model(X, y, soundList, MAX_LEN, sound_ids,
                                         primary_sounds, next_sounds, distinct_sounds,
                                         seq_len = 200,
                                         out_file = '10ep_dropout_len20.tflearn')
    f1.close()
    with open('high_10ep_drop_len20.pkl', 'wb') as f:
        pickle.dump(high, f)

    with open('mid_10ep_drop_len20.pkl', 'wb') as f:
        pickle.dump(mid, f)

    with open('low_10ep_drop_len20.pkl', 'wb') as f:
        pickle.dump(low, f)
