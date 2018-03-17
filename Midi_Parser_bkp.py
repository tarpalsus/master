# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:59:13 2018

@author: user
"""
import pretty_midi
from pretty_midi import Note
import numpy as np

NOTES_PER_OCTAVE = 12

NOTE_KEYS = [
    [0, 1, 3, 5, 7, 8, 10],
    [1, 2, 4, 6, 8, 9, 11],
    [0, 2, 3, 5, 7, 9, 10],
    [1, 3, 4, 6, 8, 10, 11],
    [0, 2, 4, 5, 7, 9, 11],
    [0, 1, 3, 5, 6, 8, 10],
    [1, 2, 4, 6, 7, 9, 11],
    [0, 2, 3, 5, 7, 8, 10],
    [1, 3, 4, 6, 8, 9, 11],
    [0, 2, 4, 5, 7, 9, 10],
    [1, 3, 5, 6, 8, 10, 11],
    [0, 2, 4, 6, 7, 9, 11]
]



class Extended_Note(pretty_midi.Note):

    def __init__(self, velocity, pitch, start, end):
        super().__init__(velocity, pitch, start, end)
        self.duration = round(self.end - self.start,2)

class MidiParser():

    def __init__(self, midi_file):
        self.midi_file = midi_file
        self.notes_with_durations = []
        self.note_names = []


    def transform(self, new_notes, instrument_type='Acoustic Grand Piano'):
        new_program = pretty_midi.instrument_name_to_program(instrument_type)
        new_instrument = pretty_midi.Instrument(new_program)
        new_file = pretty_midi.PrettyMIDI()
        new_instrument.notes = new_notes
        new_file.instruments.append(new_instrument)
        self.midi_file = new_file

    def calculate_durations(self, instrument_id = 0):
        for note in self.midi_file.instruments[instrument_id].notes:
            self.notes_with_durations.append(Extended_Note(note.velocity,
                                                           note.pitch,
                                                           note.start,
                                                           note.end))
    @property                                                      
    def get_pitches(self, instrument_id=0):
        return [x.pitch for x in self.midi_file.instruments[instrument_id].notes] 

    def quantize(self):
        for instrument in self.midi_file.instruments[:2]:
            for i, note in enumerate(instrument.notes[1:]):
                duration = round(note.end - note.start,2)
                note.start = instrument.notes[i-1].end
                note.end = note.start + duration


    def prepare_output(self, instrument_id = 0):
        notes = self.midi_file.instruments[instrument_id].notes[1:]
        for i, note in enumerate(notes):
            note.start = notes[i-1].end
            note.end = note.start + self.notes_with_durations[i].duration
        return notes


    def write(self, output_path):
        self.midi_file.write(output_path)
        
    def get_note_names(self, instrument_id=0):
        self.note_names = [x.pitch for x in self.midi_file.instruments[instrument_id].notes]
        return self.note_names
    
    def notes_histogram(self, note_pitches):
        array_notes = np.array(note_pitches)
        hist = np.bincount(array_notes % NOTES_PER_OCTAVE)
        return hist
    
    
    def get_major_key_histogram(self, note_pitches):
        """Gets a histogram of the how many notes fit into each key.
    
        Returns:
          A list of 12 ints, one for each Major key (C Major at index 0 through
          B Major at index 11). Each int is the total number of notes that could
          fit into that key.
        """
        note_histogram = self.notes_histogram(note_pitches)
        key_histogram = np.zeros(NOTES_PER_OCTAVE)
        for note, count in enumerate(note_histogram):
          key_histogram[NOTE_KEYS[note]] += count
        return key_histogram
    
    def transpose_to_key(self, key_to_transpose, notes):
        pitches = self.get_pitches
        current_key = self.get_major_key_histogram(pitches).argmax()
        transpose_amount = key_to_transpose - current_key
        new_notes = [(Note(velocity=x.velocity, 
                           pitch=x.pitch + transpose_amount,
                           start=x.start,
                           end=x.end)) for x in notes]
        return new_notes
        
        