# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:42:57 2018

@author: user
"""

import pretty_midi
midi_data = pretty_midi.PrettyMIDI(r"C:\Users\user\Downloads\bach_846.mid")
instruments = midi_data.instruments
test_filtered = pretty_midi.PrettyMIDI()
test_filtered.instruments.extend(instruments[:2])
test_filtered.write(r'C:\Users\user\Downloads\test.mid')
