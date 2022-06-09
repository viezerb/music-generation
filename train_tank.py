from gc import callbacks
from json import load
from multiprocessing import queues
from operator import index
from re import L
from statistics import mode
from music21 import converter, instrument, note, chord, stream
from pip import main
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from collections import Counter
# from tkinter import *
import os
from tqdm import tqdm
import numpy as np
import pickle
import codecs
import datetime
import tensorflow as tf
import random


dirname = '/home/music-generation'
midis_dir = os.path.join(dirname, 'EMOPIA_1.0/midis')
index_filename = 'notes_index.bin'
train_filename = 'train.bin'
model_filename = 'model.weights'
nr_of_midis = 40
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
seq_length = 10
track_length = 50
normalize = False
epochs = 100


def create_index(dirname, index_filename):
    q1_notes = []
    q2_notes = []
    q3_notes = []
    q4_notes = []

    print("Creating index")

    files = os.listdir(dirname)
    random.shuffle(files)
    for f, count in tqdm(zip(files, range(nr_of_midis))):
        #print(f)
        notes = get_notes(os.path.join(dirname, f))
        emotion = f[:2]
        if emotion == "Q1":
            q1_notes.extend(notes)
        elif emotion == "Q2":
            q2_notes.extend(notes)
        elif emotion == "Q3":
            q3_notes.extend(notes)
        elif emotion == "Q4":
            q4_notes.extend(notes)
        else:
            print("file error")
            
    q1_set = set(q1_notes)
    q2_set = set(q2_notes)
    q3_set = set(q3_notes)
    q4_set = set(q4_notes)
    q1_counted = Counter(q1_notes)
    q2_counted = Counter(q2_notes)
    q3_counted = Counter(q3_notes)
    q4_counted = Counter(q4_notes)

    q1_index = {n[1]: n[0] for n in enumerate(q1_set)}
    q2_index = {n[1]: n[0] for n in enumerate(q2_set)}
    q3_index = {n[1]: n[0] for n in enumerate(q3_set)}
    q4_index = {n[1]: n[0] for n in enumerate(q4_set)}
    print(q2_index)

    print('All different Q1 notes/chords: {}'.format(len(q1_counted)))
    print('All different Q2 notes/chords: {}'.format(len(q2_counted)))
    print('All different Q3 notes/chords: {}'.format(len(q3_counted)))
    print('All different Q4 notes/chords: {}'.format(len(q4_counted)))
    print(q2_index)

    f = codecs.open(str("q1_" + index_filename), 'wb')
    pickle.dump(q1_index, f)
    f = codecs.open(str("q2_" + index_filename), 'wb')
    pickle.dump(q2_index, f)
    f = codecs.open(str("q3_" + index_filename), 'wb')
    pickle.dump(q3_index, f)
    f = codecs.open(str("q4_" + index_filename), 'wb')
    pickle.dump(q4_index, f)
    f.close()

    return q1_counted, q2_counted, q3_counted, q4_counted


def load_index(file):
    f = codecs.open(file, 'rb')
    notes_index = pickle.load(f)
    f.close()
    return notes_index


def create_training_data(dirname, seq_length, q, notes_index, train_filename, index_count):
    X = []
    Y = []
    print("Creating training data for Q{}".format(q))
    print(notes_index)

    for f in os.listdir(dirname):
        if "Q{}_".format(q) in f:
            # print(f)
            notes = get_notes(os.path.join(dirname, f))
            # print(notes)
            seq_in = []
            seq_out = 0
            for i in range(0, len(notes) - seq_length, 1):
                if notes[i] in  notes_index:
                    if len(seq_in) < 10:
                        seq_in.append(notes[i])
                    else:
                        if normalize:
                            X.append([notes_index[n] / len(notes_index)
                                    for n in seq_in])
                        else:
                            X.append([notes_index[n] for n in seq_in])
                        seq_out = notes[i]
                        Y.append(notes_index[seq_out])
                        seq_in = []
                # seq_in = notes[i:i + seq_length]
                # seq_out = notes[i + seq_length]
                
                # print(seq_in)
    print('Q{} Training samples: {}'.format(q, len(X)))
    f = codecs.open(str("q{}_".format(q) + train_filename), 'wb')
    pickle.dump([np.array(X), np.array(Y)], f)
    f.close()


def load_training_data(file):
    f = codecs.open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data[0], data[1]


def get_notes(file):
    notes = []
    midi = converter.parse(file)

    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            #if element.octave == 5 or element.octave == 6:
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            #if all([n.octave in [4,5,6] for n in element]):
            notes.append(tuple([str(n.pitch) for n in element]))
    return notes


def create_model(X_shape, notes_index):
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, input_shape=(X_shape[1], X_shape[2]), recurrent_dropout=0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(notes_index), activation='softmax'))

    return model


def make_index():
    q1_counted, q2_counted, q3_counted, q4_counted = create_index(midis_dir, index_filename)
    q1_index = load_index(str("q1_" + index_filename))
    q2_index = load_index(str("q2_" + index_filename))
    q3_index = load_index(str("q3_" + index_filename))
    q4_index = load_index(str("q4_" + index_filename))

    create_training_data(midis_dir, seq_length, 1, q1_index, train_filename, q1_counted)
    create_training_data(midis_dir, seq_length, 2, q2_index, train_filename, q2_counted)
    create_training_data(midis_dir, seq_length, 3, q3_index, train_filename, q3_counted)
    create_training_data(midis_dir, seq_length, 4, q4_index, train_filename, q4_counted)


def train(q):
    notes_index = load_index(str("q{}_".format(q) + index_filename))
    X, Y = load_training_data(str("q{}_".format(q) + train_filename))
    print(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = to_categorical(Y, num_classes=len(notes_index))
    print(len(X))
    print(len(Y))

    model = create_model(X.shape, notes_index)
    print(model.summary)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X, Y, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])
    model.save_weights(str("q{}_".format(q) + model_filename))


def get_random_quarter(s1, s2):
    emotion = np.random.choice(np.arange(0,2), p=[s1 / 100.0, 1 - (s1 / 100.0)])
    intensity = np.random.choice(np.arange(0,2), p=[s2 / 100.0, 1 - (s2 / 100.0)])

    if emotion == 1 and intensity == 1:
        return 1
    elif emotion == 1 and intensity == 0:
        return 4
    elif emotion == 0 and intensity == 1:
        return 2
    elif emotion == 0 and intensity == 0:
        return 3
    else:
        print("Error")
        return 0

def run_generate(w1, w2):

    q1_index = load_index(str("q1_" + index_filename))
    q2_index = load_index(str("q2_" + index_filename))
    q3_index = load_index(str("q3_" + index_filename))
    q4_index = load_index(str("q4_" + index_filename))

    q1_index_inv = {i[1]: i[0] for i in q1_index.items()}
    q2_index_inv = {i[1]: i[0] for i in q2_index.items()}
    q3_index_inv = {i[1]: i[0] for i in q3_index.items()}
    q4_index_inv = {i[1]: i[0] for i in q4_index.items()}

    X_1, Y_1 = load_training_data(str("q1_" + train_filename))
    X_2, Y_2 = load_training_data(str("q2_" + train_filename))
    X_3, Y_3 = load_training_data(str("q3_" + train_filename))
    X_4, Y_4 = load_training_data(str("q4_" + train_filename))

    X_1 = np.reshape(X_1, (X_1.shape[0], X_1.shape[1], 1))
    X_2 = np.reshape(X_2, (X_2.shape[0], X_2.shape[1], 1))
    X_3 = np.reshape(X_3, (X_3.shape[0], X_3.shape[1], 1))
    X_4 = np.reshape(X_4, (X_4.shape[0], X_4.shape[1], 1))

    model_1 = create_model(X_1.shape, q1_index)
    model_2 = create_model(X_2.shape, q2_index)
    model_3 = create_model(X_3.shape, q3_index)
    model_4 = create_model(X_4.shape, q4_index)

    model_1.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model_2.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model_3.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model_4.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model_1.load_weights(str("q1_" + model_filename))
    model_2.load_weights(str("q2_" + model_filename))
    model_3.load_weights(str("q3_" + model_filename))
    model_4.load_weights(str("q4_" + model_filename))

    start = get_random_quarter(w1, w2)
    if start == 1:
        start_notes = np.random.randint(0, len(q1_index), size=seq_length)
    if start == 2:
        start_notes = np.random.randint(0, len(q2_index), size=seq_length)
    if start == 3:
        start_notes = np.random.randint(0, len(q3_index), size=seq_length)
    if start == 4:
        start_notes = np.random.randint(0, len(q4_index), size=seq_length)
    st = stream.Stream()
    offset = 0

    for i in range(track_length):
        quarter = get_random_quarter(w1, w2)
        if normalize:
            y = [model_1, model_2, model_3, model_4][quarter - 1].predict(np.reshape(
                start_notes, (1, seq_length, 1)) / len([q1_index, q2_index, q3_index, q4_index][quarter - 1]))
        else:
            y = [model_1, model_2, model_3, model_4][quarter - 1].predict(np.reshape(start_notes, (1, seq_length, 1)))

        note_num = np.argmax(y)
        note_str = [q1_index_inv, q2_index_inv, q3_index_inv, q4_index_inv][quarter - 1][note_num]
        if isinstance(note_str, str):
            note_out = note.Note(note_str)
        else:
            note_out = chord.Chord(note_str)
        note_out.offset = offset
        note_out.storedInstrument = instrument.Piano()
        st.append(note_out)
        offset += 0.5
        start_notes = np.array(list(start_notes[1:]) + [note_num])
        print(start_notes)

    st.write('midi', 'outputs/out' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.mid')
if __name__ == '__main__':
    make_index()
    train(1)
    train(2)
    train(3)
    train(4)

    #run_generate(100,100)
    #run_generate(100,0)
    #run_generate(0,100)
    #run_generate(0,0)
