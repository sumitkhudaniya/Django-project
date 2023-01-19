from django.shortcuts import render
from django.http import FileResponse, HttpResponse
from django.urls import reverse
import numpy as np
import librosa
import midiutil
import sys


# Create your views here.
def trans_matrix(note_min, note_max, p_stay_note, p_stay_silence):
    m_min = librosa.note_to_midi(note_min)
    m_max = librosa.note_to_midi(note_max)
    n_notes = m_max - m_min + 1
    p_ = (1 - p_stay_silence) / n_notes
    p__ = (1 - p_stay_note) / (n_notes + 1)
    a = np.zeros((2 * n_notes + 1, 2 * n_notes + 1))
    a[0, 0] = p_stay_silence

    for i in range(n_notes):
        a[0, (i * 2) + 1] = p_

    for i in range(n_notes):
        a[(i * 2) + 1, (i * 2) + 2] = 1

    for i in range(n_notes):
        a[(i * 2) + 2, 0] = p__
        a[(i * 2) + 2, (i * 2) + 2] = p_stay_note
        for j in range(n_notes):
            a[(i * 2) + 2, (j * 2) + 1] = p__
    return a


def prob(y, note_min, note_max, sr, frame_length, window_length, hop_length, pitch_acc, voiced_acc, onset_acc,
         spread):
    f_min = librosa.note_to_hz(note_min)
    f_max = librosa.note_to_hz(note_max)
    m_min = librosa.note_to_midi(note_min)
    m_max = librosa.note_to_midi(note_max)
    n_notes = m_max - m_min + 1
    f0, v_flag, voiced_prob = librosa.pyin(y, f_min * 0.9, f_max * 1.1, sr, frame_length, window_length, hop_length)
    tuning = librosa.pitch_tuning(f0)
    f0_ = np.round(librosa.hz_to_midi(f0 - tuning)).astype(int)
    onsets = librosa.onset.onset_detect(y, sr=sr, hop_length=hop_length, backtrack=True)
    n = np.ones((n_notes * 2 + 1, len(f0)))

    for t in range(len(f0)):
        if not v_flag[t]:
            n[0, t] = voiced_acc
        else:
            n[0, t] = 1 - voiced_acc

        for j in range(n_notes):
            if t in onsets:
                n[(j * 2) + 1, t] = onset_acc
            else:
                n[(j * 2) + 1, t] = 1 - onset_acc

            if j + m_min == f0_[t]:
                n[(j * 2) + 2, t] = pitch_acc

            elif np.abs(j + m_min - f0_[t]) == 1:
                n[(j * 2) + 2, t] = pitch_acc * spread

            else:
                n[(j * 2) + 2, t] = 1 - pitch_acc
    return n


def states_to_piano(states, note_min, note_max, hop_time):
    m_min = librosa.note_to_midi(note_min)
    states_ = np.hstack((states, np.zeros(1)))
    silence = 0
    onset = 1
    sustain = 2
    my_state = silence
    output = []
    l_onset = 0
    last_offset = 0
    l_midi = 0

    for i in range(len(states_)):
        if my_state == silence:
            if int(states_[i] % 2) != 0:
                l_onset = i * hop_time
                l_midi = ((states_[i] - 1) / 2) + m_min
                last_note = librosa.midi_to_note(l_midi)
                my_state = onset
        elif my_state == onset:
            if int(states_[i] % 2) == 0:
                my_state = sustain
        elif my_state == sustain:
            if int(states_[i] % 2) != 0:
                last_offset = i * hop_time
                my_note = [l_onset, last_offset, l_midi, last_note]
                output.append(my_note)
                l_onset = i * hop_time
                l_midi = ((states_[i] - 1) / 2) + m_min
                last_note = librosa.midi_to_note(l_midi)
                my_state = onset
            elif states_[i] == 0:
                last_offset = i * hop_time
                my_note = [l_onset, last_offset, l_midi, last_note]
                output.append(my_note)
                my_state = silence
    return output


def piano_to_midi(y, piano):
    bpm = librosa.beat.tempo(y)[0]
    print("Tempo of the original file is: " + str(bpm))
    quarter_note = 60 / bpm
    onsets = np.array([p[0] for p in piano])
    offsets = np.array([p[1] for p in piano])
    onsets = onsets / quarter_note
    offsets = offsets / quarter_note
    durations = offsets - onsets
    my_midi = midiutil.MIDIFile(1)
    my_midi.addTempo(0, 0, bpm)
    for i in range(len(onsets)):
        my_midi.addNote(0, 0, int(piano[i][2]), onsets[i], durations[i], 100)
    return my_midi


def run(wavfile, midifile):
    note_min = 'A2'
    note_max = 'E6'
    voiced_acc = 0.9
    onset_acc = 0.8
    frame_length = 2048
    window_length = 1024
    hop_length = 256
    pitch_acc = 0.99
    spread = 0.6

    y, sr = librosa.load(wavfile)

    T = trans_matrix(note_min, note_max, 0.9, 0.2)
    P = prob(y, note_min, note_max, sr, frame_length, window_length, hop_length, pitch_acc, voiced_acc,
             onset_acc, spread)
    p_init = np.zeros(T.shape[0])
    p_init[0] = 1

    states = librosa.sequence.viterbi(P, T, p_init=p_init)
    pianoroll = states_to_piano(states, note_min, note_max, hop_length / sr)
    MyMIDI = piano_to_midi(y, pianoroll)
    with open(midifile, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    return midifile


def index(request):
    if request.method == 'POST':
        input_wav = request.FILES['wav_file']
        output_file = "converted_output.mid"
        converted = run(input_wav, output_file)
        response = FileResponse(open(converted, 'rb'))
        return response
    else:
        return render(request, 'index.html')
