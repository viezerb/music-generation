from json import load
from operator import index
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
from tkinter import *
import os
from tqdm import tqdm
import numpy as np
import pickle
import codecs
import datetime
import random


dirname = '/Users/balazsviezer/code/music-generation/music-generation'
midis_dir = os.path.join(dirname, 'EMOPIA_1.0/midis')
index_filename = 'notes_index.bin'
train_filename = 'train.bin'
model_filename = 'model.weights'
nr_of_midis = 80
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
seq_length = 10
track_length = 200
normalize = False
epochs = 100


def create_index(dirname, index_filename):
    q1_notes = []
    q2_notes = []
    q3_notes = []
    q4_notes = []
    q1_files = 0
    q2_files = 0
    q3_files = 0
    q4_files = 0
    print("Creating index")

    for f, count in tqdm(zip(os.listdir(dirname), range(nr_of_midis))):
        notes = get_notes(os.path.join(dirname, f))
        emotion = f[:2]
        if emotion == "Q1":
            q1_notes.extend(notes)
            q1_files += 1
        elif emotion == "Q2":
            q2_notes.extend(notes)
            q2_files += 1
        elif emotion == "Q3":
            q3_notes.extend(notes)
            q3_files += 1
        elif emotion == "Q4":
            q4_notes.extend(notes)
            q4_files += 1
        else:
            print("file error")

    print(len(q1_notes))
    print(len(q2_notes))
    print(len(q3_notes))
    print(len(q4_notes))

    all_q1 = set(q1_notes)
    all_q2 = set(q2_notes)
    all_q3 = set(q3_notes)
    all_q4 = set(q4_notes)

    q1_index = {n[1]: n[0] for n in enumerate(all_q1) if n[0] >= 3}
    q2_index = {n[1]: n[0] for n in enumerate(all_q2) if n[0] >= 3}
    q3_index = {n[1]: n[0] for n in enumerate(all_q3) if n[0] >= 3}
    q4_index = {n[1]: n[0] for n in enumerate(all_q4) if n[0] >= 3}
    print(q1_index)
    print(len(q1_index))
    print('All different Q1 notes/chords: {}'.format(len(q1_index)))
    print('All different Q2 notes/chords: {}'.format(len(q2_index)))
    print('All different Q3 notes/chords: {}'.format(len(q3_index)))
    print('All different Q4 notes/chords: {}'.format(len(q4_index)))
    f = codecs.open(str("q1_" + index_filename), 'wb')
    pickle.dump(q1_index, f)
    f = codecs.open(str("q2_" + index_filename), 'wb')
    pickle.dump(q2_index, f)
    f = codecs.open(str("q3_" + index_filename), 'wb')
    pickle.dump(q3_index, f)
    f = codecs.open(str("q4_" + index_filename), 'wb')
    pickle.dump(q4_index, f)
    f.close()


    return q1_files, q2_files, q3_files, q4_files


def load_index(file):
    f = codecs.open(file, 'rb')
    notes_index = pickle.load(f)
    f.close()
    return notes_index


def create_training_data(dirname, seq_length, q, notes_index, train_filename, nr_of_files):
    X = []
    Y = []
    print("Creating training data for Q{}".format(q))
    print(notes_index)

    for f in os.listdir(dirname):
        if "Q{}_".format(q) in f:
            notes = get_notes(os.path.join(dirname, f))
            seq_in = []
            seq_out = 0

            for i in range(0, len(notes) - seq_length, 1):
                if len(seq_in) < 10:
                    if notes[i] in notes_index:
                        seq_in.append(notes[i])
                else:
                    if notes[i] in  notes_index:
                        seq_out = notes[i]
                        # seq_in = notes[i:i + seq_length]
                        # seq_out = notes[i + seq_length]
                        if normalize:
                            X.append([notes_index[n] / len(notes_index)
                                    for n in seq_in])
                        else:
                            X.append([notes_index[n] for n in seq_in])
                        Y.append(notes_index[seq_out])
                        seq_in = []
                        # print("X = ", len(seq_in))
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
            if element.octave == 5 or element.octave == 6:
                notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            if all([n.octave in [5,6] for n in element]):
                notes.append(tuple([str(n.pitch) for n in element]))
    return notes


def create_model(X_shape, notes_index):
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, input_shape=(
        X_shape[1], X_shape[2]), recurrent_dropout=0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(notes_index), activation='softmax'))

    return model


def make_index():
    q1_files, q2_files, q3_files, q4_files = create_index(midis_dir, index_filename)
    q1_index = load_index(str("q1_" + index_filename))
    q2_index = load_index(str("q2_" + index_filename))
    q3_index = load_index(str("q3_" + index_filename))
    q4_index = load_index(str("q4_" + index_filename))

    print("files")
    print(q1_files, q2_files, q3_files, q4_files)
    print("total")
    print(q1_files + q2_files + q3_files + q4_files)


    create_training_data(midis_dir, seq_length, 1, q1_index, train_filename, q1_files)
    create_training_data(midis_dir, seq_length, 2, q2_index, train_filename, q2_files)
    create_training_data(midis_dir, seq_length, 3, q3_index, train_filename, q3_files)
    create_training_data(midis_dir, seq_length, 4, q4_index, train_filename, q4_files)


def train(q):
    notes_index = load_index(str("q{}_".format(q) + index_filename))
    X, Y = load_training_data(str("q{}_".format(q) + train_filename))
    print("len y = ", len(Y))
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = to_categorical(Y)
    print("X.shape = ", X.shape, ", Y.shape = ", Y.shape)
    print(len(X))
    print(len(Y))

    model = create_model(X.shape, notes_index)
    print(model.summary)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X, Y, epochs=epochs, batch_size=32, validation_split=0.2)
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

    st.write('midi', 'out' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.mid')

def generate_all():
    master = Tk()
    master.resizable(width=False, height=False)
    master.geometry("250x120")
    label1 = Label(master, text="Valence")
    label1.place(x=5, y=15)
    # label1.pack()
    w1= Scale(master, from_=0, to=100, orient=HORIZONTAL)
    w1.pack()
    label2 = Label(master, text="Arousal")
    label2.place(x=5, y=56)
    w2 = Scale(master, from_=0, to=100, orient=HORIZONTAL)
    w2.pack()
    Button(master, text='Generate', command=lambda: run_generate(w1.get(), w2.get())).pack()
    mainloop()

if __name__ == '__main__':
    make_index()
    train(1)
    train(2)
    train(3)
    train(4)

    # generate_all()
