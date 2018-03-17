# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:03:42 2018

@author: user
"""
import os
import pretty_midi
import numpy as np
import h5py
BEAT_PARTS = 6
OUT_MAX_LEN = 1
NUM_CHANNELS = 4
fs=50
PITCHES = 128
PITCHES_SILENCE = PITCHES + 1
from note_events import create_events_from_midi, encode_events, MAX_LEN
from Midi_Parser import MidiParser
import random

def create_sequences(monophonic_roll, maxlen=MAX_LEN, 
                     output_maxlen=OUT_MAX_LEN, pitches=PITCHES_SILENCE):
    step=10
    num_seq = int((monophonic_roll.shape[1]-maxlen- output_maxlen)/step)
    X = np.zeros([num_seq, maxlen, pitches])
    y = np.zeros([num_seq, output_maxlen, pitches])
    for i in range(0, num_seq):
        if i==0:
            X = monophonic_roll[:,i*step:i*step+maxlen].T
            y = monophonic_roll[:,i*step+maxlen:i*step + maxlen + output_maxlen].T
        else:
            if not np.all(((monophonic_roll[128, i*step:i*step+maxlen].T) == 1)) and not np.all(((monophonic_roll[129, i*step:i*step+maxlen].T) == 1)) :
                X = np.dstack([X,monophonic_roll[:,i*step:i*step+maxlen].T])
                y = np.dstack([y, monophonic_roll[:,i*step+maxlen:i*step + maxlen + output_maxlen].T])
    X = np.moveaxis(X, -1, 0)
    y = np.moveaxis(y, -1, 0)
    return X, y

def create_sequences_fast(monophonic_roll, maxlen=MAX_LEN, output_maxlen=OUT_MAX_LEN, pitches=PITCHES_SILENCE):
    step=1
    num_seq = int((monophonic_roll.shape[1]-maxlen- output_maxlen)/step)
    X = np.zeros([num_seq, maxlen,pitches])
    y = np.zeros([num_seq, output_maxlen, pitches])
    j=0
    for i in range(0, num_seq):
        roll_part= monophonic_roll[:,i*step:i*step+maxlen].T
        #if not np.all(roll_part[:, 128]) and not np.all(roll_part[:, 129]): #erase silence only samples
        if np.sum(roll_part[:, 128]) < 20 :  
            X[j,:,:] = roll_part
            y[j,:,:] = monophonic_roll[:,i*step+maxlen:i*step + maxlen + output_maxlen].T
            j+=1
    return X[:j,:,:], y[:j,:,:]



def parse_directory(path_to_directory, file_list):
    midis = []
    for file in file_list:
        print(file)
        midi_file = pretty_midi.PrettyMIDI(os.path.join(path_to_directory,file))
        midi_obj = MidiParser(midi_file)
        midis.append(midi_obj)
    return midis, file_list[0], file

def create_dataset(midis, fs=4, poly=False):
    roll = midis[0].midi_file.get_piano_roll(fs)
    roll_ones = (roll>0).astype(float)
    rolls =roll_ones
    if poly:   
        seq = polyphonize(roll_ones)
    else:
        seq= monophonize(roll_ones)
    X, y = create_sequences_fast(seq)
    for midi_obj in midis[1:]:
        roll = midi_obj.midi_file.get_piano_roll(fs)
        roll_ones = (roll>0).astype(float)
        if poly:   
            seq = polyphonize(roll_ones)
        else:
            seq= monophonize(roll_ones)
        X_piece, y_piece = create_sequences_fast(seq)
        print(X.shape)
        X = np.concatenate((X,X_piece),axis=0)
        #rolls = np.concatenate((rolls, roll_ones),axis=1)
        
        y = np.concatenate((y,y_piece),axis=0)
    return X, y

def parse_directory_for_events(path_to_directory, fs, file_list, max_pitch=128, beat_parts=BEAT_PARTS,
                               start_file=0):
    midi_file_name = file_list[0]
    midi_file = pretty_midi.PrettyMIDI(os.path.join(path_to_directory,midi_file_name))
    events, np_events = create_events_from_midi(midi_file,fs)
    encoded = encode_events(np_events[:,:3],beat_parts=beat_parts)
    encoded2 = encode_events(np_events[:,3:6],beat_parts=beat_parts)
    encoded3 = encode_events(np_events[:,6:9],beat_parts=beat_parts)
    encoded4 = encode_events(np_events[:,9:12],beat_parts=beat_parts)
    encoded_stack = np.concatenate((encoded, encoded2, encoded3, encoded4))
    encoded_stack = encoded
    X, y = create_sequences_fast(encoded_stack, pitches=encoded_stack.shape[0])
    for midi_file_name in file_list:
        print(midi_file_name)
        midi_file = pretty_midi.PrettyMIDI(os.path.join(path_to_directory,midi_file_name))
        events_iter, np_events_iter = create_events_from_midi(midi_file,fs)
        #beat_parts = midi_file.time_signature_changes[0].numerator
        encoded_iter = encode_events(np_events_iter[:,:3],beat_parts=BEAT_PARTS)
        encoded_iter2 = encode_events(np_events_iter[:,3:6],beat_parts=BEAT_PARTS)
        encoded_iter3 = encode_events(np_events_iter[:,6:9],beat_parts=BEAT_PARTS)
        encoded_iter4 = encode_events(np_events_iter[:,9:12],beat_parts=BEAT_PARTS)
        encoded_iter_stack = np.concatenate((encoded_iter, encoded_iter2,
                                             encoded_iter3, encoded_iter4))
        encoded_iter_stack = encoded_iter
        events = events + events_iter
        np_events = np.concatenate((np_events,np_events_iter),axis=0)
        encoded_stack = np.concatenate((encoded_stack,encoded_iter_stack),axis=1)  
        print(encoded_iter_stack.shape)
        X_iter, y_iter = create_sequences_fast(encoded_iter_stack, pitches=encoded_iter_stack.shape[0])
        X = np.concatenate((X,X_iter),axis=0)
        y = np.concatenate((y,y_iter),axis=0)
        
    return events, encoded_stack, X, y

def preprocess_to_hdf5(path_to_dir, num_in_batch, fs=50, midi_num=None, data_type='roll', poly=False):
    file_list =  os.listdir(path_to_dir)
    if midi_num:
        midi_files_num = len(file_list[:midi_num])
    else:
        midi_files_num = len(file_list(path_to_dir))
    num_files_in_batch = num_in_batch
    
    random.shuffle(file_list, random.random)
    
    for i in range(0, midi_files_num, num_files_in_batch):
        first = os.listdir(path_to_dir)[i]
        last = os.listdir(path_to_dir)[i+num_files_in_batch-1]
        if data_type=='roll':
            midis, first, last = parse_directory(path_to_dir,
                                                 file_list[i*num_files_in_batch:i*num_files_in_batch + num_files_in_batch])
            X, y = create_dataset(midis, fs=fs, poly=poly)
        else:
            events, encoded, X, y = parse_directory_for_events(path_to_dir, fs,
                                                               file_list[i*num_files_in_batch:i*num_files_in_batch + num_files_in_batch])
        with h5py.File(r'C:\Users\user\Desktop\Sound_generator\processed_h5\{}_{}.h5'.format(first,last), 'w') as hf:
            hf.create_dataset("data", data=X)
            hf.create_dataset("labels", data=y)

def monophonize(piano_roll, channel=1):
    piano_roll = np.vstack([piano_roll,np.zeros((1,piano_roll.shape[1]))])
    out_roll = np.zeros(piano_roll.shape)
    prev_index = None
    for i, window in enumerate(piano_roll.T):
        if np.sum(window) ==0:
            window[window.shape[0]-1] = 1
        elif np.sum(window)>1:
            indices = np.argwhere(window == np.amax(window))
            end = indices.shape[0]
            repress = window[-2]
            window = np.zeros(window.shape)          
            if repress:
                index = indices[-1 -channel]
                window[index] = 1
                prev_index =index
            else :
                index = indices[0-channel]
                if prev_index == index:
                    window[index] = 1
                    prev_index = index
        out_roll[:,i] = window
    return out_roll

def monophonize_poly(piano_roll, channel=0, repress_value_encode=1):
    piano_roll = np.vstack([piano_roll,np.zeros((1,piano_roll.shape[1]))])
    out_roll = np.zeros(piano_roll.shape)
    prev_index = None
    for i, window in enumerate(piano_roll.T):
        if np.sum(window) ==0:
            window[window.shape[0]-1] = 1
        elif np.sum(window)>1:
            indices = np.argwhere(window == np.amax(window))
            end = indices.shape[0]
            repress = window[-2]
            window = np.zeros(window.shape) 
            try:
                if repress:
                    index = indices[channel]
                    window[index] = repress_value_encode
                    prev_index =index
                else :
                    index = indices[channel]
                    if prev_index == index:
                        window[index] = 1
                        prev_index = index
            except:
                pass
        out_roll[:,i] = window
    return out_roll

def polyphonize(piano_roll, num_channels=NUM_CHANNELS):
    piano_roll = np.vstack([piano_roll,np.zeros((1,piano_roll.shape[1]))])
    indices_list = num_channels * [piano_roll.shape[0]-1]
    indices_list = np.array(indices_list)
    piano_roll_expanded = np.tile(piano_roll, (num_channels, 1))
     
    out_roll = np.zeros(piano_roll_expanded.shape)

    for i, window in enumerate(piano_roll.T):
        window_expanded = np.zeros(window.shape[0] * NUM_CHANNELS)
        if np.sum(window) ==0:
            window_expanded[window_expanded.shape[0]-1] = 1 #Encoding of total silence, not sure if necessary
        elif np.sum(window)>=1:
            if np.sum(window) == 1:  
                indices = np.argwhere(window == np.amax(window))[:NUM_CHANNELS]
            else:
                indices = np.squeeze(np.argwhere(window == np.amax(window)))[:NUM_CHANNELS]
            #print(i, indices)
            end = indices.shape[0]
            indices_list[:end] = indices
            expanded_indices = [indice + i * (PITCHES) for i, indice in enumerate(indices_list)]
            window_expanded[expanded_indices] = 1
        out_roll[:,i] = window_expanded
    return out_roll


if __name__ =='__main__': 
    file_list =  os.listdir(r'C:\Users\user\Desktop\Sound_generator\piano_midi')[:3]
    events, encoded, X ,y = parse_directory_for_events(r'C:\Users\user\Desktop\Sound_generator\piano_midi',50, file_list)
    preprocess_to_hdf5(r'C:\Users\user\Desktop\Sound_generator\piano_midi', 1, midi_num=1, data_type='events')
