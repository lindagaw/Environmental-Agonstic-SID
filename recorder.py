#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil

import scipy.io as sio
import librosa
import soundfile as sf

# Used to record audio streams
import wave
import datetime
import time

# Used to process audio data
import contextlib
import pyaudio
from pydub import AudioSegment

# Import the voice activity detection module
import speech_recognition as sr

# disable warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def record_single_session(CHUNK, FORMAT, CHANNELS, RATE, RECORD_SECONDS, speaker_name):

    print('recording in progress ...')

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    WAVE_OUTPUT_FILENAME = speaker_name + '.wav'

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # print("Recording finished...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Generated audio file " + WAVE_OUTPUT_FILENAME)

    shutil.move(WAVE_OUTPUT_FILENAME, '1-Recording//singles//' + speaker_name + '//' + WAVE_OUTPUT_FILENAME)

    return WAVE_OUTPUT_FILENAME


def record(RECORD_SECONDS, speaker_name):

    raw_audio_dir = '1-Recording//singles//' + speaker_name + '//'
    #raw_audio_dir = 'D://raw//'
    if not os.path.isdir(raw_audio_dir):
        os.makedirs(raw_audio_dir)

    record_single_session(CHUNK, FORMAT, CHANNELS, RATE, RECORD_SECONDS, speaker_name)
    # after a window is created, signal the VAD module
