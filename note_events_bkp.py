# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:09:08 2018

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:15:00 2018

@author: user
"""

import pretty_midi
import numpy as np
import os
from midi_test import create_sequences
from Midi_Parser import MidiParser
import math
BEAT_PARTS = 6


#Jesli midi_file resolution jest 480, to znaczy, że tyle trwa cwiercnuta


#Similar function can be used for tempo / time signature quantization
def quantize_velocity(input_value, quant_value=10, max_value=128):
    max_quant = int(max_value / quant_value) + 1
    quantized_value = int(input_value / quant_value)
    encoded = np.zeros(max_quant)
    encoded[quantized_value] = 1
    return encoded

def estimate_tempo(midi_file):
    notes = midi_file.instruments[0].notes
    durations_ticks = [midi_file.time_to_tick(note.end) - \
                       midi_file.time_to_tick(note.start) for note in notes]
    durations = [note.end-note.start for note in notes]
    durations_quantized = [duration/resolution for duration in durations_ticks]
    #Tam gdzie durations_quantized = 0.25 tam jest ćwiartka. Podzielenie 60s przez
    #czas trwania tej ćwiartki da estymatę tempa

#Now just need to pass this times and tempos to event creation
def quantize_tempo_changes(midi_file, quant_step=15):
    times, tempos = midi_file.get_tempo_changes()
    tempos = np.rint((tempos)/quant_step) * quant_step
    change_indices = np.where(tempos[:-1] != tempos[1:])[0]
    return times[change_indices], tempos[change_indices]

def create_events_from_midi(midi_file, fs, instrument_id=0, beat_parts=BEAT_PARTS):
    notes = midi_file.instruments[instrument_id].notes
    beats = midi_file.get_beats()
    downbeats = midi_file.get_downbeats()
    end_of_file = (midi_file.get_end_time() +1) * fs
    note_events =[]
    note_starts = []
    note_ends = []
    note_numpy=  np.zeros((int(end_of_file),6))
    for note in notes:
        note.start = int(round(note.start *fs))
        note.end = int(round(note.end * fs))
                                                    #Monophonize part
        if note.end > note.start and note_numpy[note.start, :].all()==0:
            note.start = note.start
            note.end = note.end
            note_starts.append((note.start, note.pitch))
            note_numpy[note.start, 0] = note.pitch
            note_numpy[note.end, 1] = note.pitch
            note_numpy[note.start,5] = note.velocity
            note_ends.append((note.end, note.pitch))
            note_events.append((note.start, note.pitch, "START",note.velocity))
            note_events.append((note.end, note.pitch, "END", note.velocity))
            note_numpy[note.start+1 : note.end,2] = note.pitch
            for i in range(note.start+1,note.end,1):
                note_events.append((i, note.pitch, "HOLD", note.velocity))

    for downbeat in downbeats:
        downbeat = int(round(downbeat*fs))
        note_numpy[downbeat,4] = 1
        note_events.append((downbeat,"DOWNBEAT_START"))

    for i, beat in enumerate(beats[:-1]):
        next_beat = beats[i+1]
        for i,beat_split in enumerate(np.linspace(beat, next_beat, beat_parts, endpoint=False)):
            beat_split = int(round(beat_split*fs))
            note_numpy[beat_split,3]= i+1  #math.ceil(i+1/2) if we would like four value per beat split
        next_beat = int(round(next_beat*fs))
        beat = int(round(beat *fs))
        note_numpy[beat,3] = 1
        note_events.append((beat,"BEAT_START"))

    return note_events, note_numpy

def parse_directory_for_events(path_to_directory, max_pitch=128):
    midi_file_name = os.listdir(path_to_directory)[0]
    midi_file = pretty_midi.PrettyMIDI(os.path.join(path_to_directory,midi_file_name))
    events, np_events = create_events_from_midi(midi_file,fs)
    encoded = encode_events(np_events)

    for midi_file_name in os.listdir(path_to_directory)[1:]:
        print(midi_file_name)
        midi_file = pretty_midi.PrettyMIDI(os.path.join(path_to_directory,midi_file_name))
        events_iter, np_events_iter = create_events_from_midi(midi_file,fs)
        encoded_iter = encode_events(np_events_iter)
        events = events + events_iter
        np_events = np.concatenate((np_events,np_events_iter),axis=0)
        encoded = np.concatenate((encoded,encoded_iter),axis=1)
    return events, encoded

def encode_events(np_events, max_pitch=128, beat_parts=BEAT_PARTS, add_velocity=True):

    if add_velocity:
        encoded = np.zeros((max_pitch+4+beat_parts + quantize_velocity(128).shape[0]+1,
                            np_events.shape[0]))
        end_index = max_pitch + 4 + beat_parts + 1
    else:
        encoded = np.zeros((max_pitch+4+beat_parts, np_events.shape[0]))
    for i,event in enumerate(np_events):
        if np.all(event[:3]==0): #Encode silence
            encoded[max_pitch,i] = 1
        else:
            start_pitch = int(event[0])
            hold = int(event[2] !=0)
            if start_pitch>0: #Encode note start pitch
                encoded[start_pitch,i] = 1
            if hold !=0: #Encode note hold event
                encoded[max_pitch+1,i] = 1
            elif start_pitch == 0: #Encode note end event
                encoded[max_pitch+2,i] = 1
        encoded[max_pitch+3,i] = event[4] #Encode downbeat start
        if event[3] != 0: #Encode beat position (value of event[3] determines position)
            encoded[max_pitch+3+int(event[3]), i] = 1
        if add_velocity: #Encode quantized velocity
            encoded[end_index:,i] = quantize_velocity(event[5])
    return encoded


def decode(encoded_seq, fs, max_pitch=128,):
    curr_note = None
    notes= []
    for i,event in enumerate(encoded_seq.T):
        pitch = np.argmax(event)
        if pitch < max_pitch:
            if curr_note != None:
                curr_note.end = i/fs
                notes.append(curr_note)
            curr_note = pretty_midi.Note(velocity=120, pitch=pitch, start=i/fs, end = i/fs)
        elif pitch == (max_pitch + 1): #Hold
            curr_note = curr_note
        elif pitch == (max_pitch + 2):
            curr_note.end = i/fs
            notes.append(curr_note)
            curr_note = None
    return notes




fs=100

#C:\Users\user\Desktop\Sound_generator\inputs\bach_846.mid
midi_file = pretty_midi.PrettyMIDI(r"C:\Users\user\Downloads\Guns_n_Roses_-_November_Rain.mid")
#midi_file.time_to_tick()

new_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
new_instrument = pretty_midi.Instrument(new_program)
new_file = pretty_midi.PrettyMIDI()
#new_instrument.notes = midi_file.instruments[0].notes
#new_file.instruments.append(new_instrument)
roll = midi_file.get_piano_roll(fs)
new_file = pretty_midi.PrettyMIDI()
new_parser = MidiParser(midi_file)
new_parser.concat_instrument()
midi_file = new_parser.midi_file
concat = midi_file.instruments[0].notes
new_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
#Find out how to divide track to measures, based on time signature
beat_parts = midi_file.time_signature_changes[0].numerator
events, np_events = create_events_from_midi(midi_file,fs, beat_parts=beat_parts)
encoded = encode_events(np_events,beat_parts=beat_parts)
##transposed = np_events.T
new_notes = decode(encoded, fs)
##129 is silence, 130 is hold, 131 is end
#
#
#events, encoded = parse_directory_for_events(r'C:\Users\user\Desktop\Sound_generator\inputs')
#
#X,y = create_sequences(encoded, 25, 131)
#
#notes = midi_file.instruments[0].notes
#
new_instrument = pretty_midi.Instrument(new_program)
new_instrument.notes = new_notes
new_file.instruments.append(new_instrument)
new_file.write(r'C:\Users\user\Desktop\Sound_generator\test_quant2.mid')
#
#note_events = sorted(events,key= lambda x: x[0])
