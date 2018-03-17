# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:37:22 2018

@author: Maciek
"""
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


import pretty_midi
from pretty_midi import note_number_to_name, note_name_to_number
import numpy as np
#import tflearn
import tensorflow as tf
import random
import os
import time
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Bidirectional
from keras.layers import LSTM, Input, merge, multiply
from keras.layers.core import Permute, Reshape, Flatten, Lambda, RepeatVector
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.models import load_model
import keras
from Midi_Parser import MidiParser
from keras import backend as K
from models import prepare_attention, prepare_model_keras, build_and_run_model, \
discriminator, adversarial, generator, prepare_conv_lstm, simple

from seq2seq_models import seq2seq_model, simple_seq2seq_model, attention_seq2seq_model
from vae import simple_encoder, vae_encoder
import h5py
from preprocessing import polyphonize, monophonize, create_sequences, \
create_dataset, preprocess_to_hdf5, parse_directory, parse_directory_for_events, \
monophonize_poly
from note_events import note_events_to_midi, MAX_LEN
from keras.utils import plot_model
from matplotlib import pyplot as plt

import random
import itertools

OUT_MAX_LEN = 1
PITCHES = 128
PITCHES_SILENCE = PITCHES + 1
NUM_CHANNELS = 4
#INPUT_DIM = NUM_CHANNELS * (PITCHES+2) Poly
INPUT_DIM =PITCHES_SILENCE


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
            seed = seed[:, 1:, :]
            seed = np.append(seed, pred, axis=1)
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
        low = chain_predictions(model, seed, 0.1, seq_len, poly)
        mid = chain_predictions(model, seed, 0.5, seq_len, poly)
        high = chain_predictions(model, seed, 1, seq_len, poly)
        temperatures_low.append(low)
        temperatures_mid.append(mid)
        temperatures_high.append(high)
        seeds.append(seed)
        print(i)
    print('Sequences generated')
    return seeds, temperatures_high, temperatures_low, temperatures_mid




def glue_polyphony(polyphonic_roll, num_channels = NUM_CHANNELS):
    output = polyphonic_roll[:PITCHES,:]
    for i in range(1,num_channels):
        output = output + polyphonic_roll[(PITCHES) * i:(PITCHES) * (i+1),:]
    return output
    

def piano_roll_to_midi(windows, fs,
                           instrument_name='Acoustic Grand Piano',
                           allow_represses=False):

    midi = pretty_midi.PrettyMIDI()
    
    instruments = []
    for i in range(NUM_CHANNELS):
        instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
        instrument = pretty_midi.Instrument(program=instrument_program, name=str(i))
        instruments.append(instrument)
    cur_notes = [None] * NUM_CHANNELS # an invalid note to start with
    cur_note_starts = [None] * NUM_CHANNELS
    clock = 0
    notes = []
    for i, step in enumerate(windows):
        #repress = step[PITCHES].astype(bool)
        repress = False
        note_nums = np.argwhere(step == np.amax(step))[:NUM_CHANNELS]
        if not np.all(step==0):
            note_nums = np.argwhere(step == np.amax(step))[:NUM_CHANNELS]
        else:
            note_nums = [None] * 4
        for i, note_num in enumerate(note_nums):
            if repress or note_num != cur_notes[i]:
                if cur_notes[i] is not None and cur_notes[i] >= 5 and cur_notes[i]<127:
                    note = pretty_midi.Note(velocity=127,
                                            pitch=int(cur_notes[i]),
                                            start=cur_note_starts[i],
                                            end=clock)
                    notes.append(note)
                    instruments[i].notes.append(note)
                cur_notes[i] = note_num
                cur_note_starts[i] = clock
        clock = clock + 1.0 / fs
    midi.instruments = instruments

    return midi, notes


def piano_roll_to_midi_mono(windows, fs,
                           instrument_name='Acoustic Grand Piano',
                           allow_represses=False):

    midi = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=instrument_program)
    cur_note = None # an invalid note to start with
    cur_note_start = None
    clock = 0
    list_of_processed = []
    for step in windows:
        note_num = np.argmax(step) - 1
        if allow_represses or note_num != cur_note:
            if cur_note is not None and cur_note >= 0 and cur_note <127:
                note = pretty_midi.Note(velocity=80,
                                        pitch=int(cur_note),
                                        start=cur_note_start,
                                        end=clock)
                list_of_processed
                instrument.notes.append(note)
            cur_note = note_num
            cur_note_start = clock
        clock = clock + 1.0 / fs
    midi.instruments.append(instrument)

    return midi




def transform_to_midi(x, name, poly=False, fs=96):
    x = np.array(x)   
    x = np.squeeze(x)
    if poly:
        glued = glue_polyphony(x)
    else:
        glued = x
    glued = glued[:,:128]
    glued = (glued>0).astype(float)
    midi_obj_from_roll = piano_roll_to_midi(glued, fs)
    midi_obj_from_roll.write(r'C:\Users\user\Desktop\Sound_generator\midis\{}.mid'.format(name))
    return glued


def reshape_seq(x):
    length = len(x)
    x = np.array(x)
    x = x.reshape(length, 129)
    x = x[:,:128]
    return x




def find_represses(roll):
    roll2 = np.roll(roll, 1)
    roll2[:,0]=np.zeros(roll2[:,0].shape)
    diff = roll - roll2
    small_diff=diff * (abs(diff)<30).astype(float)
    ticks = np.argwhere(diff>0)[:,1]
    roll = np.vstack((roll, np.zeros((1,roll.shape[1]))))
    roll[PITCHES, ticks] = 1
    return roll
    
def combine_history(histories):
    combined = {}
    for value in ['val_loss', 'loss', 'acc', 'val_acc']:
        combined[value]= [history[value] for history in histories]
        combined[value] = list(itertools.chain.from_iterable(combined[value]))
    return combined

def train_from_h5(h5_files,model):
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs/midis", 
                                              write_graph=True, 
                                              write_images=True)
    histories = []
    for file in h5_files:
        with h5py.File(file, 'r') as hf:
            X = hf['data'][:]
            y = np.squeeze(hf['labels'][:])
            history = model.fit(X,y, epochs=3, batch_size=128, 
                                callbacks = [tensorboard], validation_split=0.2)
            histories.append(history.history)
    return model, X, combine_history(histories)

def train_from_batch(path_to_dir, num_in_batch, model, fs=50, midi_num=None, data_type='roll', poly=False):
    start = time.time()
    if midi_num:
        midi_files_num = len(os.listdir(path_to_dir)[:midi_num])
    else:
        midi_files_num = len(os.listdir(path_to_dir))
    file_list = os.listdir(path_to_dir)
    random.shuffle(file_list, random.random)
    num_files_in_batch = num_in_batch
    histories = []
    for i in range(0, midi_files_num, num_files_in_batch):
        if data_type=='roll':
            midis, first, last = parse_directory(path_to_dir,
                                                 file_list[i*num_files_in_batch:i*num_files_in_batch + num_files_in_batch])
            X, y = create_dataset(midis, fs=fs, poly=poly)
        else:
            events, encoded, X, y = parse_directory_for_events(path_to_dir, fs,
                                                               file_list[i*num_files_in_batch:i*num_files_in_batch + num_files_in_batch])
        print(start-time.time())
        history = model.fit(X,np.squeeze(y), epochs=10, batch_size=128, validation_split=0.2)
        histories.append(history.history)
    return model,X, combine_history(histories)


def vis(history, save_path):
    print(history.keys())
# summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(save_path+'_accuracy.png')
    # summarize history for loss
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(save_path+'_loss.png')


#for path in list:
if __name__ == '__main__':
    single = True
    fs = 100
    poly = False
    start = time.time()
    seq_len = 100
    if single:
        midi_file = pretty_midi.PrettyMIDI(r"C:\Users\user\Desktop\Sound_generator\piano_midi\chpn_op25_e1.mid")
        midi_obj = MidiParser(midi_file)
        notes2 = midi_obj.get_note_names()
        roll = midi_obj.midi_file.get_piano_roll(fs)
        roll_repress = find_represses(roll)
        roll_ones = (roll_repress>0).astype(float)
    
        sums = np.sum(roll_ones,axis=0)
        midi_obj.write(r'C:\Users\user\Desktop\Sound_generator\test_no_dur.mid')
        monophonic = monophonize(roll_ones)
        poly = np.zeros(monophonic.shape)
        for i in range(4):
            monophonic1= monophonize_poly(roll_ones,i)
            channel_roll = monophonic1[:128,:] * roll
            channel_rep1 = (find_represses(channel_roll)>0).astype(float)
            monophonic1 = monophonize_poly(channel_rep1,0,repress_value_encode=0)
            poly +=monophonic
        monophonic2= monophonize_poly(roll_ones,1)
        monophonic3= monophonize_poly(roll_ones,2)
        monophonic4= monophonize_poly(roll_ones,3)
        poly = monophonic1 + monophonic2 + monophonic3+monophonic4
    
        midi_obj_from_roll = piano_roll_to_midi_mono(monophonic.T, fs)
        #midi_obj_from_roll, notes = piano_roll_to_midi(roll_ones.T, fs)
        #notes = midi_obj_from_roll.instruments[0].notes
        #notes = midi_obj.get_pitches
        #hist = midi_obj.get_major_key_histogram(notes)
        #output = midi_obj.prepare_output()
        #midi_obj.transform(output)
        midi_obj_from_roll2, notes2 = piano_roll_to_midi(poly.T, fs)
        midi_obj_from_roll2.write(r'C:\Users\user\Desktop\Sound_generator\test_dur2.mid')
        midi_obj_from_roll.write(r'C:\Users\user\Desktop\Sound_generator\test_dur.mid')
#        notes_class = midi_obj.midi_file.instruments[0].notes
        #X, y = create_sequences(monophonic)
        #X2 = np.repeat(X[:, :, :, np.newaxis, np.newaxis], 4, axis=3)
    else:
        start = time.time()
        path_to_directory = r'C:\Users\user\Desktop\Sound_generator\piano_midi'
        num_in_batch=10
        midi_num = 20
        #preprocess_to_hdf5(path_to_directory, num_in_batch,fs, midi_num, data_type='roll')
        model = simple(MAX_LEN, input_dim=129)
        files = os.listdir(r"C:\Users\user\Desktop\Sound_generator\processed_h5")
        h5_files = [r"C:\Users\user\Desktop\Sound_generator\processed_h5" + "\\"+file for file in files]
        #model, X, history = train_from_h5(h5_files[:2], model)
        model_path = r"C:\Users\user\Desktop\Sound_generator\models\simple"
       

#        model = load_model(model_path+'.h5')
        
        model, x,history = train_from_batch(path_to_directory,num_in_batch, model, midi_num=20)
        end = time.time() - start
        print(end)
        model.save(model_path+".h5")
        plot_model(model, to_file=model_path+'.png')
        vis(history, model_path)
        #seeds, high, low, mid = generate(X, model, MAX_LEN, seq_len, poly, iters=20)
#        all_notes = []
#        for i,melody in enumerate(mid):
#            try:
#                new_notes = note_events_to_midi(np.squeeze(np.array(melody).T),'gen_mid_events{}.mid'.format(i), fs=50)
#                all_notes.append(new_notes)
#            except:
#                print('Wrong midi created for {}'.format(i))
        #for i,melody in enumerate(high):
         #   transform_to_midi(melody,'gen_high_{}'.format(i), poly, fs)
#        for i,melody in enumerate(low):
#            transform_to_midi(melody,'gen_low_{}'.format(i), poly, fs)
            #X, y = create_dataset(midis, fs=fs, poly=poly)

            
#    print("Dataset generated")
#    print(start - time.time())
#    start_learning = time.time()
#    #
#    keras = True
#    if not keras:
#        model = build_and_run_model(MAX_LEN )
#        model.fit(X, y, validation_set=0.1, batch_size=128, n_epoch=8,
#                          run_id='Midis')
#    else:
#        SAVE = True
#        LOAD = False
#        model_path = 'lstm.h5'
#        if LOAD:
#            model = load_model(model_path)
#        else:
#            GAN = False
#            encoder = False
#            training_steps = 350
#            generator_model = None
#            adversarial_model = None
#            discriminator_model = None
#            batch_size = 128
#            if GAN:
#                for i in range(training_steps):
#                    x = X[i*batch_size:(i+1)*batch_size,:,:]
#                    size = x.shape[0]
#                    
#                    if not discriminator_model:
#                        discriminator_model = discriminator()
#                    if not generator_model:
#                        generator_model = generator()
#                    if not adversarial_model:
#                        adversarial_model = adversarial(generator_model, discriminator_model)
#                    noise = np.random.uniform(0, 1.0, size=[batch_size, 100])
#                    fakes = generator_model.predict(noise)
#                   
#                    x = np.concatenate((x, fakes))
#                    x = np.expand_dims(x,axis=3)
#                    y = np.zeros([2*batch_size, 1])
#                    y[:batch_size, :] = 1
#                    d_loss = discriminator_model.train_on_batch(x, y)
#                    y = np.ones([batch_size, 1])
#                    noise = np.random.uniform(0, 1.0, size=[batch_size, 100])
#                    
#                    #a_loss = adversarial_model.fit(noise, y, epochs=15)
#                    a_loss = adversarial_model.train_on_batch(noise, y)
#    
#                    log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
#        
#                    log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
#        
#                    print(log_mesg)
#            elif encoder:
#                encoder, decoder, autoencoder = simple_encoder(input_len = MAX_LEN,
#                                       input_dim = INPUT_DIM, latent_dim = 5)
#                                       #intermediate_dim=10, epsilon_std=1.0)
#                autoencoder.fit(X, X, epochs=5, batch_size=128)
#                seed = select_random_seed(X,MAX_LEN)
#                pred = autoencoder.predict(seed)
#                representation = encoder.predict(seed)
#                pred2 = decoder.predict(representation)
#            else:
#                seq2=False
#                if seq2:
#                    #X_swap = np.swapaxes(X, 1, 2)
#                    model = attention_seq2seq_model(input_dim = INPUT_DIM, hidden_dim=5,
#                                      output_dim=INPUT_DIM, output_length=OUT_MAX_LEN,
#                                      input_length = MAX_LEN)
#                    #y = np.swapaxes(y, 1, 2)
#                else:
#                    y = np.squeeze(y)
#                    #y2 = np.squeeze(y2)
#                    model = prepare_model_keras(MAX_LEN)
#                    model2 = prepare_conv_lstm(MAX_LEN, NUM_CHANNELS)
#                y2= np.repeat(y[:, :, np.newaxis, np.newaxis], 4, axis=2)
#                #model.fit(X,y,batch_size=128,epochs=3, validation_split=0.1)
#                model2.fit(X2,y2,batch_size=128,epochs=10, validation_split=0.1)
#                
#                seed = select_random_seed(X, MAX_LEN)
#                pred = model.predict(seed)
#                seed = np.repeat(seed[:, :, :, np.newaxis, np.newaxis], 4, axis=3)
#                pred2 = model2.predict(seed)
#                if SAVE:
#                    model.save(model_path)
#    #
#        print('Trained')
#        print(start_learning - time.time())
#        seeds, high, low, mid = generate(X, model, MAX_LEN, seq_len, poly, iters=20)
#        for i,melody in enumerate(mid):
#            glued = transform_to_midi(melody,'gen_mid_{}'.format(i), poly, fs)
#        for i,melody in enumerate(high):
#            transform_to_midi(melody,'gen_high_{}'.format(i), poly, fs)
#        for i,melody in enumerate(low):
#            transform_to_midi(melody,'gen_low_{}'.format(i), poly, fs)
    
    #main()
    #high = high.reshape([high.shape[1],high.shape[2]])
    
    #midi_obj_from_roll =  piano_roll_to_midi(high.T, fs)
