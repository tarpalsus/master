# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:56:27 2018

@author: user
"""

import keras
import numpy as np
from keras.layers import Input, LSTM, RepeatVector, Dense, Lambda, TimeDistributed
from keras.models import Model
from keras.regularizers import l1
from keras import backend as K
from keras import metrics, objectives

def simple_encoder(input_len, input_dim, latent_dim):
    inputs = Input(shape=(input_len, input_dim))
    encoded = LSTM(latent_dim)(inputs)
    
    decoded = RepeatVector(input_len)(encoded)
    decoded = LSTM(input_dim, return_sequences=True,
                   activity_regularizer=l1(1e-5))(decoded)
    decoded = TimeDistributed(Dense(input_dim, activation='softmax'))(decoded)
    
    
    sequence_autoencoder = Model(inputs, decoded)
    sequence_autoencoder.summary()
    encoder = Model(inputs, encoded)
    
    encoded_input = Input(shape=(latent_dim,))

    #decoder = LSTM(input_dim, return_sequences=True)(encoded_input)
    decoder_layer = sequence_autoencoder.layers[-2]
    repeat_layer = sequence_autoencoder.layers[-3]
    time_dist_layer = sequence_autoencoder.layers[-1]
    repeated = repeat_layer(encoded_input)
    output = decoder_layer(repeated)
    
    decoder = Model(encoded_input, time_dist_layer(output))
    decoder.summary()
    sequence_autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
    
    
    
    return encoder, decoder, sequence_autoencoder


epsilon_std=1.0
latent_dim=5
def sampling(args):
    z_mean, z_log_var= args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon



def vae_encoder(input_len, input_dim, latent_dim, intermediate_dim, epsilon_std):
    inputs = Input(shape=(input_len, input_dim))
    encoded = LSTM(intermediate_dim)(inputs)
    #W pracy z enkoderem, z LSTMa zwracana jest sekwencja, a z_mean i z_log_var
    #to warstwy TimeDistributed, czyli latent_vector to sekwencja, a nie tylko pojedynczy
    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)
    z = Lambda(sampling)([z_mean, z_log_var])
    decoder_h = Dense(intermediate_dim, activation='relu')
    repeated = RepeatVector(input_len)
    decoder_mean = LSTM(input_dim, return_sequences=True, activation='softmax')
    #decoder_mean = Dense(input_dim, activation='softmax')
    h_decoded = decoder_h(z)
    h_repeated = repeated(h_decoded)
    x_decoded_mean = decoder_mean(h_repeated)
    vae = Model(inputs, x_decoded_mean)
    def vae_loss(inputs, x_decoded_mean):
        inputs = K.flatten(inputs)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = input_dim * metrics.binary_crossentropy(inputs, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = xent_loss + kl_loss
        return vae_loss
    
    #vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.summary()
    
    encoder = Model(inputs, z_mean)
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _h_repeated = repeated(_h_decoded)
    _x_decoded_mean = decoder_mean(_h_repeated)
    generator = Model(decoder_input, _x_decoded_mean)
    
    return  encoder, generator, vae