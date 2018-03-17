# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 10:43:02 2018

@author: user
"""

import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from keras.layers import TimeDistributed, Dense, Activation
from keras.models import Sequential
#Seq2seq model
def simple_seq2seq_model(input_dim, hidden_dim, output_length, output_dim, input_length):  
    model = Sequential()
    model_seq = SimpleSeq2Seq(input_dim=input_dim, hidden_dim=hidden_dim,
                          output_length=output_length, output_dim=output_dim,
                          input_length=input_length, readout_activation='softmax')
    model.add(model_seq)
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def seq2seq_model(input_dim, hidden_dim, output_length, output_dim, input_length): 
    model = Sequential()
    model_seq = Seq2Seq(input_dim=input_dim, hidden_dim=hidden_dim,
                          output_length=output_length, output_dim=output_dim,
                          input_length=input_length)
    model.add(model_seq)
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def attention_seq2seq_model(input_dim, hidden_dim, output_length, output_dim, input_length): 
    model = Sequential()
    model_seq = AttentionSeq2Seq(input_dim=input_dim, hidden_dim=hidden_dim,
                          output_length=output_length, output_dim=output_dim,
                          input_length=input_length)
    model.add(model_seq)
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model