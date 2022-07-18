import tensorflow as tf
from tensorflow.keras import backend as K
import glob
import random
import pretty_midi
import IPython
import numpy as np
from tqdm import tnrange, tqdm_notebook, tqdm
from random import shuffle, seed
import numpy as np
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Nadam
import numpy as np
from numpy.random import choice
import pickle
import matplotlib.pyplot as plt

import unicodedata
import re
import numpy as np
import os
import io
import time

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):

    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI() #container for midi data
    instrument = pretty_midi.Instrument(program=program)

    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    velo_changes = np.nonzero(np.diff(piano_roll).T) #keep track of velocfity changes
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velo_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time /= fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def create_model(seq_len, unique_notes, dropout=0.2, output_emb=100, rnn_u=128, dense_u=64):
  inputs = tf.keras.layers.Input(shape=(seq_len,))
  embedding = tf.keras.layers.Embedding(input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
  fp = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_u, return_sequences=True))(embedding)
  fp = tf.keras.layers.Dropout(dropout)(fp)
  fp = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_u, return_sequences=True))(fp)
  fp = tf.keras.layers.Dropout(dropout)(fp)
  fp = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_u))(fp)
  fp = tf.keras.layers.Dropout(dropout)(fp)
  fp = tf.keras.layers.Dense(dense_u)(fp)
  fp = tf.keras.layers.LeakyReLU()(fp)
  outputs = tf.keras.layers.Dense(unique_notes+1, activation='softmax')(fp)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generate_scores_rnn')
  return model

model = create_model(seq_len, unique_notes)