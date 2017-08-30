# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:20:56 2017


"""
import pandas as pd
import guitarpro as g
import re
import guitarpro.models as gm
import guitarpro.base
from urllib.request import urlopen
import pickle

string_mapper = {1: 'e',
                 2: 'h',
                 3: 'g',
                 4: 'd',
                 5: 'a',
                 6: 'ee',
                 }
inverse_mapper = {v: k for k, v in string_mapper.items()}

value_duration_mapper = {64: 1,
                         32: 1,
                         16: 2,
                         8: 4,
                         4: 8,
                         2: 16,
                         1: 32}


class mock(object):
    """Wrap value into object, to bypass guitapro lib limitations"""
    def __init__(self, value):
        self.value = value


def from_guitar_pro(path, filter_tracks=['guitar', 'hammet', 'kirk'],
                    selected_tracks=None):
    """Get useful sounds"""
    song = guitarpro.parse(path)
    tracks = song.tracks
    if not selected_tracks:
        if not filter_tracks:
            selected = tracks
        else:
            selected = []
            for ft in filter_tracks:
                selected.extend(list(filter(lambda x: ft in x.name.lower(),
                                            tracks)))
    else:
        selected = list(filter(lambda x: x.name in selected_tracks, tracks))

    print('Parsing ' + str(len(selected)) + ' tracks')
    out = []
    grouped_tracks = pd.DataFrame()
    for guitar in selected:
        track_sounds = []
        for measure in guitar.measures:
                for beat in measure.voices[0].beats:
                    for note in beat.notes:
                        track_sounds.append((note.value, note.string,
                                    beat.start, beat.duration.value))
        df = pd.DataFrame(track_sounds,
                          columns=['Value', 'String', 'Start', 'Duration'])

        df['String'] = df['String'].map(string_mapper)
        df['Value+String'] = df['Value'].apply(str) + df['String']

        grouped_notes = df[['Value+String', 'Start']].groupby('Start').sum()
        grouped_durations = df[['Duration', 'Start']].groupby('Start').first()

        grouped_sounds = pd.concat([grouped_notes, grouped_durations], axis=1)
        grouped_sounds = grouped_sounds.reset_index(drop=True)
        out.extend(track_sounds)
        grouped_tracks = pd.concat([grouped_tracks, grouped_sounds])
    print('Parsed')
    return song, grouped_tracks.reset_index(), out


def calculate_measures(durations):
    durations_inv = [value_duration_mapper[x] for x in durations]
    indexes = []
    summed = 0
    for i, duration in enumerate(durations_inv):
        summed = summed + duration
        if summed >= 32:
            summed = 0
            indexes.append(i)
    return indexes


def to_guitar_pro(df, path='tarpalsus.g5', tempo=130):
    """ Write df to g5 file"""
    values = [tuple(x) for x in df[['Value+String', 'Duration']].values]
    splits = []
    for val in values:
        splits.append(((re.findall('\d*\D+', val[0])), val[1]))

    base_song = gm.Song(tracks=[], tempo=tempo)
    track = gm.Track(base_song)
    durations = list(zip(*splits))[1]
    indexes = calculate_measures(durations)
    chunks = []
    shifted = indexes.copy()
    indexes.insert(0, 0)
    shifted.append(len(splits))
    for index, shift in zip(indexes, shifted):
        chunks.append([splits[index:shift]])

    for i, chunk in enumerate(chunks):
        new_measure = gm.Measure(track,
                                 gm.MeasureHeader(number=i, start=i*1080,
                                                  timeSignature=gm.TimeSignature(numerator=4,
                                                  denominator=mock(4))
                                                  , tempo=180))
        voice = new_measure.voices[0]

        for i, (notes, duration) in enumerate(chunk[0]):
            new_duration = gm.Duration(value=duration)
            new_beat = gm.Beat(voice,
                               duration=new_duration,
                               status=gm.BeatStatus.normal)

            for note in notes:
                value=int(''.join(re.findall('[0-9]+', note)))
                string = int(inverse_mapper[''.join(re.findall('[a-z]+', note))])
                new_note = gm.Note(new_beat,
                                   value=value,
                                   string=string,
                                   type=gm.NoteType.normal)
                new_beat.notes.append(new_note)
            voice.beats.append(new_beat)
        track.measures.append(new_measure)
    base_song.tracks.append(track)
    g.write(base_song, path)

    return splits, base_song, chunks


def rnn_to_guitar_pro(high_path, mid_path, low_path,
                      output_name, duration=False):
    """ Get output produced by neural net, process it to g5.files"""
    with open(high_path, 'rb') as f:
        high = pickle.load(f)
    with open(low_path, 'rb') as f:
        low = pickle.load(f)
    with open(mid_path, 'rb') as f:
        mid = pickle.load(f)
    process_sequence(output_name, mid, 'mid',duration)
    process_sequence(output_name, high, 'high', duration)
    process_sequence(output_name, low, 'low', duration)
    return


def process_sequence(output_name, sequence, sequence_name, duration=False):
    for i, seq in enumerate(sequence):
        df = pd.DataFrame()
        if duration:
            print(i)
            duration = [re.split('(\d+)', i)[-2] for i in seq]
            value_string = [''.join(re.split('(\d+)', i)[1:-2]) for i in seq]
            df['Duration'] = duration
            df['Duration'] = df['Duration'].apply(int)
            df['Value+String'] = value_string
        else:
            df['Value+String'] = seq
            df['Duration'] = 8
        _, _, _ = to_guitar_pro(df, ('generated\\' + output_name + '_' + sequence_name + str(i) + '.gp5'))
    return

if __name__ == '__main__':

    paths_killem = [r"Killem\Metallica - Blitzkrieg (guitar pro).gp3",
             r"Killem\Metallica - Metal Militia (guitar pro).gp3",
             r"Killem\Metallica - Jump In The Fire (guitar pro).gp4",
             r"Killem\Metallica - Motorbreath (guitar pro).gp3",
             r"CKillem\Metallica - No Remorse (guitar pro).gp3",
             r"Killem\Metallica - Phantom Lord (guitar pro).gp3",
             r"Killem\Metallica - Seek And Destroy (guitar pro).gp5",
             r"Killem\Metallica - Whiplash (guitar pro).gp3"
             ]
    tracks_killem = [['Kirk Hammet'],
              ['Gtr. II','Gtr. III'],
              ['Guitar 1','Guitar 2'],
              ['Gtr. I Rhytm','Gtr. III Lead'],
              ['Gtr. I','Gtr. III'],
              ['Gtr. II', 'Gtr. III', 'Gtr. IV'],
              ['Gtr. I','Gtr. III'],
              ['Guitar 2', 'Guitar 3']]

    tracks_killem_riff = [['Kirk Hammet'],
              ['Gtr. II'],
              ['Guitar 1'],
              ['Gtr. I Rhytm'],
              ['Gtr. I'],
              ['Gtr. II', 'Gtr. III'],
              ['Gtr. I'],
              ['Guitar 2']]

    paths_ride = [r"ride\Metallica - Creeping Death (guitar pro).gp5",
                  r"ride\Metallica - Escape (guitar pro).gp4",
                  r"ride\Metallica - Fade To Black (guitar pro).gp5",
                  r"ride\Metallica - Fight Fire With Fire (guitar pro).gp5",
                  r"ride\Metallica - For Whom The Bell Tolls (guitar pro).gp3",
                  r"ride\Metallica - Ride The Lightning (guitar pro).gp5",
                  r"ride\Metallica - The Call Of Ktulu (guitar pro).gp3",
                  r"ride\Metallica - Trapped Under Ice (guitar pro).gp3"]
    tracks_ride = [['Kirk Hammet (Guitar)','Lead Guitar (Kirk)'],
                    ['Guitar-1','Guitar-2'],
                    ['Guitare 1 ','Guitare 2 ', 'Guitare 3 '],
                    ['Kirk Hammett'],
                    ['Kirk Hammett'],
                    ['Hammet','Lead 1', 'Lead 2'],
                    ['Gtr. I', 'Gtr. II'],
                    ['Guitar 1', 'Guitar 3']]

    tracks_ride_riff = [['Kirk Hammet (Guitar)'],
                    ['Guitar-1'],
                    ['Guitare 1 ','Guitare 2 '],
                    ['Kirk Hammett'],
                    ['Kirk Hammett'],
                    ['Hammet'],
                    ['Gtr. I'],
                    ['Guitar 1']]


    paths_master = [r"master\Metallica - Battery (guitar pro).gp3",
                    r"master\Metallica - Damage Inc (guitar pro).gp4",
                    r"master\Metallica - Disposable Heroes (guitar pro).gp5",
                    r"master\Metallica - Leper Messiah (guitar pro).gp4",
                    r"master\Metallica - Master Of Puppets (guitar pro).gp5",
                    r"master\Metallica - Orion (guitar pro).gp4",
                    r"master\Metallica - Welcome Home Sanitarium (guitar pro).gp4"
                    ]
    tracks_master = [['Gtr. 1', 'Gtr. 3'],
                     ['Guitar 2', 'Guitar 2'],
                     ['Guitar 1 (Distortion)', 'Lead Guitar 1 (Distortion)'],
                     ['Rhythm/Lead'],
                     ['James Hetfield', 'Solo', 'Clean Guitar'],
                     ['Kirk (Rhythm Guitar)', 'Kirk (Lead Guitar)'],
                     ['Rhythm', 'Lead']]

    tracks_master_riff = [['Gtr. 1'],
                     ['Guitar 2'],
                     ['Guitar 1 (Distortion)'],
                     ['Rhythm/Lead'],
                     ['James Hetfield'],
                     ['Kirk (Rhythm Guitar)'],
                     ['Rhythm', 'Lead']]

    paths_justice = [r"justice\Metallica - Blackened (guitar pro).gp4",
                     r"justice\Metallica - Dyers Eve (guitar pro).gp4",
                     r"justice\Metallica - Eye Of The Beholder (guitar pro).gp4",
                     r"justice\Metallica - Harvester Of Sorrow (guitar pro).gp4",
                     r"justice\Metallica - One (guitar pro).gp5",
                     r"justice\Metallica - The Frayed Ends Of Sanity (guitar pro).gp4",
                     r"justice\Metallica - The Shortest Straw (guitar pro).gp5",
                     r"justice\Metallica - To Live Is To Die (guitar pro).gp4",
                     r"justice\metallica-and_justice_for_all.gp4"]

    tracks_justice = [['James Hetfield (Guitar I)','Guitar solo'],
                       ['Guitar 1' , 'Guitar 3'],
                       ['Dist Gtr 1', 'Solo Gr'],
                       ['Kirk'],
                       ['James (TREBBLE)','Kirk (BASS)','Aux/Solo Gtr1'],
                       ['Guitar 2', 'Guitar 3'],
                       ['Guitar 1 (Distortion)','Lead Guitar 1 (Distortion)',
                        'Lead Guitar 2 (Distortion)'],
                       ['Acoustic Guitar','Kirk Hammet', '4th guitar', '5th guitar'],
                       ['Guitar I', 'Guitar III', 'Guitar V']
                       ]

    tracks_justice_riff = [['James Hetfield (Guitar I)'],
                       ['Guitar 1' ],
                       ['Dist Gtr 1'],
                       ['Kirk'],
                       ['Kirk (BASS)'],
                       ['Guitar 2'],
                       ['Guitar 1 (Distortion)'],
                       ['Kirk Hammet'],
                       ['Guitar I', 'Guitar III']
                       ]


    paths_black = [r"black\Metallica - Dont Tread On Me (guitar pro).gp5",
                   r"black\Metallica - Enter Sandman (guitar pro).gp5",
                   r"black\Metallica - Holier Than Thou (guitar pro).gp5",
                   r"black\Metallica - My Friend Of Misery (guitar pro).gp4",
                   r"black\Metallica - Nothing Else Matters (guitar pro).gp4",
                   r"black\Metallica - Of Wolf And Man (guitar pro).gp3",
                   r"black\Metallica - The God That Failed (guitar pro).gp3",
                   r"black\Metallica - The Struggle Within (guitar pro).gp4",
                   r"black\Metallica - The Unforgiven (guitar pro).gp5",
                   r"black\Metallica - Through The Never (guitar pro).gp4",
                   r"black\Metallica - Wherever I May Roam (guitar pro).gp4"]

    tracks_black = [['Rhythm 1','Solo'],
                    ['James Rythm 1','Kirk Lead 1'],
                    ['Guitar II - Kirk','Guitar V - Solo'],
                    ['Guitar Clean/Dist (solo I)','Guitar Dist (solo I)','Distortion Guitar 2'],
                    ['Electro-Acoustic Guitar (James Hetfield)', 'Electric Guitar (Kirk Hammett)'],
                    ['Guitar 1', 'Guitar Solo'],
                    ['Gtr. 1','Gtr. 3'],
                    ['Guitar 1','Kirk Hammet Solo'],
                    ['James - Acoustic','Kirk - Distortion'],
                    ['Guitar I - Rythm Guitar','Guitar III - Lead Guitar'],
                    ['Rhy Gtr','Lead Gtr']]


    tracks_black_riff = [['Rhythm 1'],
                    ['James Rythm 1'],
                    ['Guitar II - Kirk'],
                    ['Distortion Guitar 2'],
                    ['Electric Guitar (Kirk Hammett)'],
                    ['Guitar 1'],
                    ['Gtr. 1'],
                    ['Guitar 1'],
                    ['Kirk - Distortion'],
                    ['Guitar I - Rythm Guitar'],
                    ['Rhy Gtr']]

    def album_process(paths, tracks, outfile):
        """Take all above listed tracks and write them to csv, for learning"""
        sounds = pd.DataFrame()

        for path, track in zip(paths, tracks):
            base_song, sounds_out, out = from_guitar_pro(path,
                                                 track)
            sounds = pd.concat([sounds, sounds_out])
        sounds.to_csv(outfile)

rnn_to_guitar_pro('high_no_dur_10ep.pkl',
                  'mid_no_dur_10ep.pkl', 'low_no_dur_10ep.pkl', '10ep_no_dur')
rnn_to_guitar_pro('high_10ep.pkl',
                  'mid_10ep.pkl', 'low_10ep.pkl', '10ep', duration = True)
rnn_to_guitar_pro('high_10ep_drop.pkl',
                  'mid_10ep_drop.pkl', 'low_10ep_drop.pkl', '10ep_drop', duration = True)
rnn_to_guitar_pro('high_10ep_drop_len70.pkl',
                  'mid_10ep_drop_len70.pkl', 'low_10ep_drop_len70.pkl', '10ep_drop_len70', duration = True)
rnn_to_guitar_pro('high_10ep_drop_len5.pkl',
                  'mid_10ep_drop_len5.pkl', 'low_10ep_drop_len5.pkl', '10ep_drop_len5', duration = True)
rnn_to_guitar_pro('high_10ep_drop_len20.pkl',
                  'mid_10ep_drop_len20.pkl', 'low_10ep_drop_len20.pkl', '10ep_drop_len20', duration = True)