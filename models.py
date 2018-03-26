# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:10:20 2018

@author: user
"""
import tflearn
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Bidirectional
from keras.layers import LSTM, Input, merge, multiply, Conv2D, \
 Conv2DTranspose, BatchNormalization, UpSampling2D, ConvLSTM2D, Conv3D, BatchNormalization
from keras.layers.core import Permute, Reshape, Flatten, Lambda, RepeatVector
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.data_utils import get_file
from keras.models import load_model
from Midi_Parser import MidiParser
from keras import backend as K
MAX_LEN = 25
OUT_MAX_LEN = 1
from preprocessing import PITCHES_REPRESS
NUM_CHANNELS = 4
INPUT_DIM = NUM_CHANNELS * PITCHES_REPRESS
INPUT_DIM = PITCHES_REPRESS

#Regular models
def simple(maxlen, input_dim=INPUT_DIM):
    model = Sequential()
#    model.add(LSTM(512, input_shape=(maxlen, input_dim), return_sequences=True))
#    model.add(Dropout(0.5))
#    model.add(LSTM(512, input_shape=(maxlen, input_dim),
#                  return_sequences=True, dropout_U=0.5))
#    model.add(Dropout(0.5))
    model.add(LSTM(512, input_shape=(maxlen, input_dim), dropout_U=0.5))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.001, clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def prepare_model_keras(maxlen, input_dim=INPUT_DIM):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, input_dim), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, input_shape=(maxlen, input_dim),
                  return_sequences=True, dropout_U=0.5))
    model.add(Dropout(0.5))
    model.add(LSTM(128, input_shape=(maxlen, input_dim), dropout_U=0.5))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    return model

def prepare_conv_lstm(maxlen, num_channels):
    model = Sequential()
    model.add(ConvLSTM2D(filters=1, kernel_size=(10,10), input_shape=(None, INPUT_DIM, num_channels,1),
                         return_sequences=True, padding='same'))
    model.add(ConvLSTM2D(filters=1, kernel_size=(10,10),
                   padding='same'))
    #model.add(BatchNormalization())
    model.add(Conv2D(filters=1, kernel_size=(10, 10),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model



#Attention building functions
def prepare_attention(maxlen, full_attention=False):
    inputs = Input(shape=(maxlen, INPUT_DIM,))
    lstm_units = 32
    if full_attention: #Means using state_h in each step
        lstm_out, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(inputs)
        state_c = RepeatVector(MAX_LEN)(state_c)
        lstm_out = merge([lstm_out, state_c], mode='sum')
    else:
        lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out, full_attention)
    #attention_mul = Flatten()(attention_mul)
    attention_mul2 = attention_3d_block(attention_mul, full_attention)
    attention_mul2= Lambda(lambda x: K.sum(x, axis=1))(attention_mul2)
    output = Dense(INPUT_DIM, activation='softmax', name='output')(attention_mul2)
    model = Model(input=[inputs], output=output)
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def attention_3d_block(inputs, full_attention):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, MAX_LEN))(a)
    if full_attention:
        a = Dense(MAX_LEN, activation='tanh')(a)
    a = Dense(MAX_LEN, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([inputs, a_probs])
    #output_attention_mul = merge([inputs, a_probs], mode='mul')
    return output_attention_mul



#GAN Model
def discriminator(maxlen=MAX_LEN, depth = 64):
    model = Sequential()
#    model.add(Conv2D(depth*1, 5, strides=2,
#                     padding='same', activation='relu',  input_shape=(MAX_LEN, INPUT_DIM,1)))
#    model.add(Dropout(0.9))
#    model.add(Conv2D(depth*2, 5, strides=2,
#                     padding='same', activation='relu'))
#    model.add(Dropout(0.9))
#    model.add(Conv2D(depth*4, 5, strides=2,
#                     padding='same', activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Conv2D(depth*8, 5, strides=2,
#                     padding='same', activation='relu'))

    model.add(Flatten(input_shape=(MAX_LEN, INPUT_DIM,1)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    optimizer = Adam(.002, 0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,\
    metrics=['accuracy'])
    return model


def adversarial(generator, discriminator):
    optimizer = Adam(0.0002, 0.5)
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,\
    metrics=['accuracy'])
    model.summary()
    return model

def generator(maxlen = MAX_LEN):
    optimizer = Adam(0.0002, 0.5)
    noise_shape = (100,)
    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(MAX_LEN * INPUT_DIM, activation='tanh'))
    model.add(Reshape((MAX_LEN,INPUT_DIM)))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer,\
    metrics=['accuracy'])
    return model



#Tflearn variant
def build_and_run_model(X, maxlen,
                        seq_len=50, out_file='model.tflearn'):
    """ Build and create sequences with different temperatures (measure of "innovation")"""
    with tf.Graph().as_default():
        net = tflearn.input_data([None, maxlen, 129],
                                  data_preprocessing=None,
                                  data_augmentation=None)
        net = tflearn.lstm(net, 512, return_seq=True)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.lstm(net, 512, return_seq=True)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.lstm(net, 512)
        net = tflearn.fully_connected(net, X.shape[2],
                                      activation='softmax')
        net = tflearn.regression(net, optimizer='adam',
                                 loss='categorical_crossentropy',
                                 learning_rate=0.001)
        model = tflearn.DNN(net, clip_gradients=5.0)


    return model