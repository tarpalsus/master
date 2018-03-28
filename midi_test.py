# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:37:22 2018

@author: Maciek
"""
#Wnioski:
#Zmiana stepu na wiekszy dobrze wplywa na proces uczenia, szybciej zbiega, val_loss tez zatrzymuje sie na
#nizszym poziomie




import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from sklearn.model_selection import train_test_split
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

#from seq2seq_models import seq2seq_model, simple_seq2seq_model, attention_seq2seq_model
from vae import simple_encoder, vae_encoder
import h5py
from preprocessing import polyphonize, monophonize, create_sequences, \
create_dataset, preprocess_to_hdf5, parse_directory, parse_directory_for_events, \
monophonize_poly, find_represses, squeeze_roll, expand_roll, \
 PITCHES, PITCHES_SILENCE, PITCHES_REPRESS
from note_events import note_events_to_midi, MAX_LEN
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.utils import resample

import random
import itertools

OUT_MAX_LEN = 1
NUM_CHANNELS = 4
#INPUT_DIM = NUM_CHANNELS * (PITCHES+2) Poly
INPUT_DIM =PITCHES_REPRESS




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
    for step in windows:
        note_num = np.argmax(step) - 1
        if note_num != cur_note:
            if cur_note is not None and cur_note >= 0 and cur_note <127:
                note = pretty_midi.Note(velocity=80,
                                        pitch=int(cur_note),
                                        start=cur_note_start,
                                        end=clock)
                instrument.notes.append(note)
            cur_note = note_num
            cur_note_start = clock
        clock = clock + 1.0 / fs
    midi.instruments.append(instrument)

    return midi




def transform_to_midi(x, name, dir_path=r'C:\Users\user\Desktop\Sound_generator\midis\{}.mid', poly=False, fs=96):
    x = np.array(x)
    x = np.squeeze(x)
    if poly:
        glued = glue_polyphony(x)
    else:
        glued = x
    glued = glued[:,:128]
    glued = (glued>0).astype(float)
    glued[-1,:] = np.zeros(128)
    midi_obj_from_roll = piano_roll_to_midi_mono(glued, fs)
    midi_obj_from_roll.write(dir_path.format(name))
    return glued


def reshape_seq(x):
    length = len(x)
    x = np.array(x)
    x = x.reshape(length, 129)
    x = x[:,:128]
    return x






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
    if midi_num:
        midi_files_num = len(os.listdir(path_to_dir)[:midi_num])
    else:
        midi_files_num = len(os.listdir(path_to_dir))
    file_list = os.listdir(path_to_dir)
    random.shuffle(file_list, random.random)

    num_files_in_batch = num_in_batch
    histories = []
    for i in range(0, int(midi_files_num/num_files_in_batch)-1):
        if data_type=='roll':
            midis, first, last = parse_directory(path_to_dir,
                                                 file_list[i*num_files_in_batch:i*num_files_in_batch + num_files_in_batch])
            X, y = create_dataset(midis, fs=fs, poly=poly)
        else:
            events, encoded, X, y = parse_directory_for_events(path_to_dir, fs,
                                                               file_list[i*num_files_in_batch:i*num_files_in_batch + num_files_in_batch])
        print(start-time.time())
        history = model.fit(X,np.squeeze(y), epochs=5, batch_size=128, validation_split=0.2)
        histories.append(history.history)
    X=None
    y=None
    return model, history, combine_history(histories)

def train_from_generator(path_to_dir, batch_size=128, val_split=0.5):
    file_list = os.listdir(path_to_dir)[111:117]
    midis, first, last = parse_directory(path_to_dir,
                                                 file_list)
    #piano midi for fs=50 has 3827803 sequences in total /512 -> 7500 steps
    DATA_LEN = 3827803
    DATA_LEN = 400 * len(file_list)
    midis = midis
    random.shuffle(midis,random.random)
    steps_per_epoch = int((DATA_LEN / batch_size) * (1-val_split))
    print(steps_per_epoch)
    val_steps_per_epoch = int((DATA_LEN / batch_size) * (val_split))
    print(val_steps_per_epoch)
    history= model.fit_generator(generator=sampleGenerator(midis,batch_size=batch_size),
                                 epochs=1000, steps_per_epoch=steps_per_epoch,
                                 validation_data=sampleGenerator(midis, batch_size=batch_size, shuffle_piece=False, train=False),
                                 validation_steps=val_steps_per_epoch)
    return model, history

def sampleGenerator(midis, batch_size, fs=50, shuffle_piece=False, train=True):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset

          # Generate batches
          #random.shuffle(midis,random.random)
          for i in range(int(len(midis)/3)):
              # Find list of IDs
              #midis_temp = midis[i*batch_size:(i+1)*batch_size]
              # Generate data
              X, y = create_dataset(midis[i:i+3], fs=fs, poly=False)
              #X_train, X_test, y_train, y_test = train_test_split(X, y,
               #                                                   test_size=0.2,
                #                                                  random_state=7)

              

              imax = int(X.shape[0]/batch_size) - 1
              indexes = np.arange(imax)

              randomize = np.arange(len(X))
              np.random.seed(0)
              np.random.shuffle(randomize)
              X = X[randomize]
              y = y[randomize]
              if train:
                  X = X[:int(0.8*len(X))]
                  y = y[:int(0.8*len(y))]
                  for j in indexes:
                      yield X[j*batch_size:(j+1)*batch_size,:,:], np.squeeze(y[j*batch_size:(j+1)*batch_size,:,:])
              else:
                  X = X[int(0.8*len(X)):]
                  y = y[int(0.8*len(y)):]
                  yield X, np.squeeze(y)
              if shuffle_piece:
                  random.shuffle(indexes)
              


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
    single = False
    fs = 50
    poly = False
    generator = True
    start = time.time()
    seq_len = 100
    if single:
        midi_file = pretty_midi.PrettyMIDI(r"C:\Users\user\Desktop\Sound_generator\piano_midi\bach_846.mid")
        #midi_file= pretty_midi.PrettyMIDI(r"C:\Users\Maciek\Downloads\inputs\bach_846.mid")
        midi_obj = MidiParser(midi_file)
        notes2 = midi_obj.get_note_names()
        roll = midi_obj.midi_file.get_piano_roll(fs)
        squeezed = squeeze_roll(roll)
        #squeezed = roll
        roll_repress = find_represses(squeezed)
        roll_ones = (roll_repress>0).astype(float)

        sums = np.sum(roll_ones,axis=0)
        #midi_obj.write(r'C:\Users\user\Desktop\Sound_generator\test_no_dur.mid')
        repress = False
        monophonic = monophonize(roll_ones, delete_repress=repress)

        poly = np.zeros(monophonic.shape)
        for i in range(4):
            monophonic1= monophonize_poly(roll_ones,i)
            channel_roll = monophonic1[:-2,:] * squeezed
            channel_rep1 = (find_represses(channel_roll)>0).astype(float)
            monophonic1 = monophonize_poly(channel_rep1,0,repress_value_encode=0)
            poly +=monophonic
        monophonic2= monophonize_poly(roll_ones,1)
        monophonic3= monophonize_poly(roll_ones,2)
        monophonic4= monophonize_poly(roll_ones,3)
        poly = monophonic1 + monophonic2 + monophonic3+monophonic4
        monophonic_unsq = expand_roll(monophonic, delete_repress=repress)
        midi_obj_from_roll = piano_roll_to_midi_mono(monophonic_unsq.T, fs)
        #midi_obj_from_roll, notes = piano_roll_to_midi(roll_ones.T, fs)
        #notes = midi_obj_from_roll.instruments[0].notes
        #notes = midi_obj.get_pitches
        #hist = midi_obj.get_major_key_histogram(notes)
        #output = midi_obj.prepare_output()
        #midi_obj.transform(output)
        midi_obj_from_roll2, notes2 = piano_roll_to_midi(poly.T, fs)
        #midi_obj_from_roll2.write(r'C:\Users\user\Desktop\Sound_generator\test_dur2.mid')
        midi_obj_from_roll.write(r'C:\Users\user\Desktop\Sound_generator\test_dur.mid')
        #midi_obj_from_roll.write(r'C:\Users\Maciek\Downloads\master-master\test_dur.mid')
#        notes_class = midi_obj.midi_file.instruments[0].notes
        #X, y = create_sequences(monophonic)
        #X2 = np.repeat(X[:, :, :, np.newaxis, np.newaxis], 4, axis=3)
    elif generator:
        start = time.time()
        path_to_directory = r'C:\Users\user\Desktop\Sound_generator\piano_midi'
        #path_to_directory = r"C:\Users\Maciek\Downloads\inputs"
        num_in_batch=7
        midi_num = None
        #preprocess_to_hdf5(path_to_directory, num_in_batch,fs, midi_num, data_type='roll')
        model = simple(MAX_LEN, input_dim=INPUT_DIM)
        #model = prepare_model_keras(MAX_LEN, input_dim=INPUT_DIM)
        files = os.listdir(r"C:\Users\user\Desktop\Sound_generator\processed_h5")
        h5_files = [r"C:\Users\user\Desktop\Sound_generator\processed_h5" + "\\"+file for file in files]
        #model, X, history = train_from_h5(h5_files[:2], model)
        model_path = r"C:\Users\user\Desktop\Sound_generator\models\test"


#        model = load_model(model_path+'.h5')
        histories = []
        #model,history = train_from_batch(path_to_directory,num_in_batch, model)
        model,history = train_from_generator(path_to_directory)
        history = history.history
#        for i in range(5):
#            model,history = train_from_batch(path_to_directory,num_in_batch, model)
#            histories.append(history)
        end = time.time() - start
        print(end)
        #history = combine_history(histories)

        model.save(model_path+".h5")
        plot_model(model, to_file=model_path+'.png')
        vis(history, model_path)
    else:
        #path_to_directory = r"C:\Users\Maciek\Downloads\inputs"
        path_to_directory=r"C:\Users\user\Desktop\Sound_generator\test"
        file_list = os.listdir(path_to_directory)
        midis, first, last = parse_directory(path_to_directory,
                                                 file_list)
        X,y = create_dataset(midis,fs)
#    print("Dataset generated")
#    print(start - time.time())
#    start_learning = time.time()
#    #
        keras = True
        if not keras:
            model = build_and_run_model(MAX_LEN )
            model.fit(X, y, validation_set=0.1, batch_size=128, n_epoch=8,
                              run_id='Midis')
        else:
            SAVE = True
            LOAD = False
            model_path = 'lstm_repress_filtered.h5'
            if LOAD:
                model = load_model(model_path)
            else:
                GAN = False
                encoder = False
                training_steps = 350
                generator_model = None
                adversarial_model = None
                discriminator_model = None
                batch_size = 128
                if GAN:
                    for i in range(training_steps):
                        x = X[i*batch_size:(i+1)*batch_size,:,:]
                        size = x.shape[0]

                        if not discriminator_model:
                            discriminator_model = discriminator()
                        if not generator_model:
                            generator_model = generator()
                        if not adversarial_model:
                            adversarial_model = adversarial(generator_model, discriminator_model)
                        noise = np.random.uniform(0, 1.0, size=[batch_size, 100])
                        fakes = generator_model.predict(noise)

                        x = np.concatenate((x, fakes))
                        x = np.expand_dims(x,axis=3)
                        y = np.zeros([2*batch_size, 1])
                        y[:batch_size, :] = 1
                        d_loss = discriminator_model.train_on_batch(x, y)
                        y = np.ones([batch_size, 1])
                        noise = np.random.uniform(0, 1.0, size=[batch_size, 100])

                        #a_loss = adversarial_model.fit(noise, y, epochs=15)
                        a_loss = adversarial_model.train_on_batch(noise, y)

                        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])

                        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])

                        print(log_mesg)
                elif encoder:
                    encoder, decoder, autoencoder = simple_encoder(input_len = MAX_LEN,
                                           input_dim = INPUT_DIM, latent_dim = 5)
                                           #intermediate_dim=10, epsilon_std=1.0)
                    autoencoder.fit(X, X, epochs=5, batch_size=128)
                    seed = select_random_seed(X,MAX_LEN)
                    pred = autoencoder.predict(seed)
                    representation = encoder.predict(seed)
                    pred2 = decoder.predict(representation)
                else:
                    seq2=False
                    if seq2:
                        #X_swap = np.swapaxes(X, 1, 2)
                        model = attention_seq2seq_model(input_dim = INPUT_DIM, hidden_dim=5,
                                          output_dim=INPUT_DIM, output_length=OUT_MAX_LEN,
                                          input_length = MAX_LEN)
                        #y = np.swapaxes(y, 1, 2)
                    else:
                        y = np.squeeze(y)
                        #y2 = np.squeeze(y2)
                        model = simple(MAX_LEN)
                        #model = prepare_model_keras(MAX_LEN)
                        model2 = prepare_conv_lstm(MAX_LEN, NUM_CHANNELS)
                    y2= np.repeat(y[:, :, np.newaxis, np.newaxis], 4, axis=2)
                    randomize = np.arange(len(X))
                    np.random.shuffle(randomize)
                    X = X[randomize]
                    y = y[randomize]
                    model.fit(X,y,batch_size=128,epochs=100, validation_split=0.1)
                    #model2.fit(X2,y2,batch_size=128,epochs=10, validation_split=0.1)

                    #seed = select_random_seed(X, MAX_LEN)
#                    pred = model.predict(seed)
#                    seed = np.repeat(seed[:, :, :, np.newaxis, np.newaxis], 4, axis=3)
#                    pred2 = model2.predict(seed)
                    if SAVE:
                        model.save(model_path)
        #
#            print('Trained')
#            print(start_learning - time.time())
#            seeds, high, low, mid = generate(X, model, MAX_LEN, seq_len, poly, iters=20)
#            for i,melody in enumerate(mid):
#                glued = transform_to_midi(melody,'gen_mid_{}'.format(i), poly, fs)
#            for i,melody in enumerate(high):
#                transform_to_midi(melody,'gen_high_{}'.format(i), poly, fs)
#            for i,melody in enumerate(low):
#                transform_to_midi(melody,'gen_low_{}'.format(i), poly, fs)
#
#        main()
#        high = high.reshape([high.shape[1],high.shape[2]])
#
#        midi_obj_from_roll =  piano_roll_to_midi(high.T, fs)
