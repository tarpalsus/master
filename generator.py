# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:54:57 2018

@author: user
"""

import numpy as np
import pretty_midi
from Midi_Parser import MidiParser
from midi_test import NUM_CHANNELS, PITCHES, MAX_LEN, note_events_to_midi, transform_to_midi
import random
from keras.models import load_model
from preprocessing import create_dataset, expand_roll

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = preds.flatten()
    probas = np.random.multinomial(1, preds)
    return np.argmax(probas)

def chain_predictions(model, seed, temperature, seq_len, poly=False):
    output = []
    for i in range(seq_len):
        pred = model.predict(seed)
        if poly:
            pred_out = np.zeros(pred.shape)
            if len(pred_out.shape) ==2:
                pred_out = pred_out.reshape([pred_out.shape[0],1,pred_out.shape[1]])
            for i in range(NUM_CHANNELS):
                pred_channel = pred[:, i * PITCHES : (i+1) * PITCHES]
                index = sample(pred_channel)
                pred_channel = np.zeros(pred_channel.shape)
                if len(pred.shape) ==2:
                    pred_channel = pred_channel.reshape([pred_channel.shape[0],1,pred_channel.shape[1]])
                pred_channel[:, :, index] = 1.0
                pred_out[:,:,i * PITCHES : (i+1) * PITCHES] = pred_channel
            seed = seed[:, 1:, :]
            seed = np.append(seed, pred_out, axis=1)
                #np.append(output, pred, axis=1)
            output.append(pred_out)
        else:
            index = sample(pred, temperature)
            pred = np.zeros(pred.shape)
            if len(pred.shape) ==2:
                pred = pred.reshape([pred.shape[0],1,pred.shape[1]])
            pred[:, :, index] = 1.0
            seed = np.roll(seed,-1 ,axis=1)
            seed[:,MAX_LEN-1,:] = pred
            #np.append(output, pred, axis=1)
            output.append(pred)
    return output

def select_random_seed(X, maxlen):
    start_index = random.randint(0, len(X) - maxlen - 1)
    seed = X[start_index: start_index + 1,:,:]
    return seed





def generate(X, model, maxlen, seq_len, poly, iters):
    temperatures_low = []
    temperatures_high = []
    temperatures_mid = []
    seeds = []
    for i in range(iters):
        seed = select_random_seed(X, maxlen)
        #generated = gen(model, seed, maxlen, maxlen)
        low = chain_predictions(model, seed, 0.3, seq_len, poly)
        mid = chain_predictions(model, seed, 0.7, seq_len, poly)
        high = chain_predictions(model, seed, 1, seq_len, poly)
        temperatures_low.append(low)
        temperatures_mid.append(mid)
        temperatures_high.append(high)
        seeds.append(seed)
        print(i)
    print('Sequences generated')
    return seeds, temperatures_high, temperatures_low, temperatures_mid


if __name__ =='__main__':
    #model_path = r"C:\Users\user\Desktop\Sound_generator\models\simple_quick.h5"
    model_path = r"C:\Users\Maciek\Downloads\master-master\master-master\lstm_repress_filtered.h5"
    #seed_file = r"C:\Users\user\Desktop\Sound_generator\piano_midi\bach_846.mid"
    seed_file = r"C:\Users\Maciek\Downloads\inputs\bach_846.mid"
    out_path=r"C:\Users\Maciek\Downloads\master-master\master-master\{}.mid"
    fs=50
    seq_len = 1000
    midi_file = pretty_midi.PrettyMIDI(seed_file)
    midi_obj = MidiParser(midi_file)
    X,y = create_dataset(midi_obj, fs=fs)
    poly=False

    model = load_model(model_path)
    seeds, high, low, mid = generate(X, model, MAX_LEN, seq_len, poly, iters=5)
    all_notes = []
#    for i,melody in enumerate(mid):
#        try:
#            new_notes = note_events_to_midi(np.squeeze(np.array(melody).T),'gen_mid_events{}.mid'.format(i), fs=50)
#            all_notes.append(new_notes)
#        except:
#            print('Wrong midi created for {}'.format(i))
    for i,melody in enumerate(high):
        x = np.array(melody)
        x = expand_roll(np.squeeze(x).T, delete_repress=True).T
        seed = expand_roll(np.squeeze(seeds[i]).T, delete_repress=True).T
        combine = np.concatenate((seed,x))
        transform_to_midi(x,'gen_high_new{}'.format(i), out_path, poly, fs)
    for i,melody in enumerate(mid):
        x = np.array(melody)
        x = expand_roll(np.squeeze(x).T, delete_repress=True).T
        seed = expand_roll(np.squeeze(seeds[i]).T, delete_repress=True).T
        combine = np.concatenate((seed,x))
        transform_to_midi(x,'gen_mid_{}'.format(i), out_path,  poly, fs)