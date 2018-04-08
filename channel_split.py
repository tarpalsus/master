# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 12:34:57 2018

@author: Maciek
"""
import numpy as np
from preprocessing import squeeze_roll, find_represses, polyphonize, monophonize_poly, \
create_sequences_fast
NUM_CHANNELS = 3

def split_to_channels(roll, channels=NUM_CHANNELS):
    output_rolls = []
    for channel in range(channels):
        out_roll = np.zeros(roll.shape)
        for i, window in enumerate(roll):
            indices = np.where(window>0)
            values = window[indices]
            window = np.zeros(window.shape)
            try:
                index = indices[0][channel]
                window[index] = values[channel]
            except IndexError:
                pass
            out_roll[i,:] = window
        output_rolls.append(out_roll.T)
    return output_rolls

#out =split_to_channels(roll.T)
def create_dataset_channels(midis, fs=50, concat=True):
    if isinstance(midis,list):
        roll = midis[0].midi_file.get_piano_roll(fs)
        roll = squeeze_roll(roll)
        channels = split_to_channels(roll.T,NUM_CHANNELS)

        X_channels = []
        y_channels = []
        for i, channel in enumerate(channels):
            roll_repress = find_represses(channel)
            roll_ones = (roll_repress>0).astype(float)
            monophonic = monophonize_poly(roll_ones, i+1,0)
            X, y = create_sequences_fast(monophonic)
            X_channels.append(X)
            y_channels.append(y)
        X_channels = np.array(X_channels)
        y_channels = np.array(y_channels)
        #print(X.shape)
        for midi_obj in midis[1:]:
            roll = midi_obj.midi_file.get_piano_roll(fs)
            roll = squeeze_roll(roll)
            channels = split_to_channels(roll.T,NUM_CHANNELS)
            X_channels_piece = []
            y_channels_piece = []
            for i, channel in enumerate(channels):
                roll_repress = find_represses(channel)
                roll_ones = (roll_repress>0).astype(float)
                monophonic = monophonize_poly(roll_ones, i+1,0)
                X_piece, y_piece = create_sequences_fast(monophonic)
                X_channels_piece.append(X_piece)
                y_channels_piece.append(y_piece)
            X_channels_piece = np.array(X_channels_piece)
            y_channels_piece = np.array(y_channels_piece)
            X_channels = np.concatenate((X_channels,X_channels_piece),axis=1)
            #rolls = np.concatenate((rolls, roll_ones),axis=1)
            #print(X.shape)
            y_channels = np.concatenate((y_channels,y_channels_piece),axis=1)

    if concat:
        X = X_channels
        y = y_channels
        X = X.reshape((X.shape[0]*X.shape[1],X.shape[2],X.shape[3]))
        y = y.reshape((y.shape[0]*y.shape[1],y.shape[2],y.shape[3]))
    return X, y


def fold(array):
    return array.reshape((NUM_CHANNELS, int(array.shape[0] / NUM_CHANNELS),
                          array.shape[1], array.shape[2]))