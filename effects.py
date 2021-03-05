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

from audio import Audio
# from separate import Separate

class Effects(Audio):
    def __init__(self, path, filename, rate, out_harm, out_perc, out_bass):
        '''__init__ function.
            :parameters:
                - path : str
                    absolute path to input audio
                - filename : str
                    name of songs in directory and type (pop song.wav, jazz song.mp3, etc)
                - rate : int
                    Sample rate of each song
                - out_harm : str
                    path to output of harmonic seperation
                - out_perc : str
                    path to output of percussive seperation
                - out_bass : str
                    path to output of bass seperation
        '''
        super().__init__(path, filename, rate, out_harm, out_perc, out_bass)
        # input_song = self.input_song
        # stft_song = self.stft_song

    def load_input(self):
        '''
            load_input function.
        '''
        print('Loading First song..', self.filename)
        y, self.rate = librosa.load(self.path)
        return y

    def calc_stft(self):
        '''
            calc_stft function
        '''
        y = self.load_input()
        FRAME_SIZE = 2048
        HOP_SIZE = 512
        print('Extract STFT...')
        D = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        return D

    def amp_to_db(self):
        '''
            amp_to_db function
        '''
        #D = self.calc_stft()
        S_db = librosa.power_to_db(np.abs(self.stft_song)**2)
        return S_db
    
    def plot_input_spectrogram(self):
        '''
            plot_spectogram fuction that plots the Spectrogram of the input audio file
        '''
        S_db = self.amp_to_db()
        plt.figure(figsize=(8, 4))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(S_db, y_axis='log')
        plt.colorbar(format="%+2.f")
        plt.title("Spectogram")
        plt.tight_layout()
        plt.show()
