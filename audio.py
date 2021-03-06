'''
    imported Libraries
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import scipy
import librosa
import librosa.display
from playsound import playsound
from pydub import AudioSegment
import os
import soundfile as sf
from scipy.signal import butter, lfilter

class Audio:
    
    def __init__(self, path, filename, rate, out_harm, out_perc, out_bass, out_vocals, input_song=None):
        '''
        __init__ function.

            :Args:
                - path (str) : absolute path to input audio.
                - filename (str) : name of songs in directory and type (pop song.wav, jazz song.mp3, etc).
                - rate (float) : Sample rate of each song.
                - out_harm (str) : path to output of harmonic seperation.
                - out_perc (str) : path to output of percussive seperation.
                - out_bass (str) : path to output of bass seperation.
                - out_vocals (str) : path to output of vocals seperation.
                - input_song (ndarray) : numpy array of the loaded input song.
        '''
        self.path = path
        self.filename = filename
        self.rate = rate
        self.out_harm = out_harm
        self.out_perc = out_perc
        self.out_bass = out_bass
        self.out_vocals = out_vocals
        self.input_song = input_song
    
    def __str__(self):
        return f"File path {self.path} \nFilename {self.filename} \nSample rate {self.rate} \nOutput path of harmonics {self.out_harm}\nOutput path of percussive {self.out_perc}\nOutput path of bass {self.out_bass}\nOutput path of vocals {self.out_vocals}"
