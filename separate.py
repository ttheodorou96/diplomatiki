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

from effects import Effects

class Separate(Effects):
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
        #y_harmonic = self.stft_harmonic

    def hpss_demo(self):  
        D = self.calc_stft()
        # Separate components with the effects module
        print('Separating harmonics and percussives... ')
        y_harmonic, y_percussive = librosa.decompose.hpss(D, margin=4.0) #type complex64 must istst to save them

        # --------test Types of numpy arrays---------#
        # print(D.shape)
        # print(D.dtype)
        # print(y_harmonic.shape)
        # print(y_harmonic.dtype)
        return y_harmonic, y_percussive

    def invert_stft(self):
        y = self.hpss_demo()
        y_harmonic = y[0]
        y_percussive = y[1]
        print('Separation Complete!\nApply Inverted STFT, ready to save results in wav form... ')
        out_harmonic = librosa.core.istft(y_harmonic) #type float32 can be used in sf.write.
        out_percussive = librosa.core.istft(y_percussive)

        # --------test Types of numpy arrays---------#
        # print(out_harmonic.shape)
        # print(out_harmonic.dtype)
        return out_harmonic, out_percussive

    def butter_lowpass(self,cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self,data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def get_outputs(self):

        # Filter requirements for bass isolation.
        order = 6
        fs = self.rate      # sample rate, Hz
        cutoff = 250  # desired cutoff frequency of the filter, Hz

        # Get the filter coefficients.
        b, a = self.butter_lowpass(cutoff, fs, order)
        output = self.invert_stft()

        out_harmonic = output[0]
        out_percussive = output[1]
        out_bass = self.butter_lowpass_filter(out_harmonic, cutoff, fs, order)

        return out_harmonic, out_percussive, out_bass

    def save_output(self):
        out = self.get_outputs()
        out_harmonic = out[0]
        out_percussive = out[1]
        out_bass = out[2]

        # Save the results
        output_harmonic = os.path.join(self.out_harm, 'output_harmonic.wav') #path of out_harm
        print('Saving harmonic audio to: ', output_harmonic)
        sf.write(file=output_harmonic, data=out_harmonic, samplerate=self.rate)
        
        output_percussive = os.path.join(self.out_perc, 'output_percussive.wav') #path of perc
        print('Saving percussive audio to: ', output_percussive)
        sf.write(file=output_percussive, data=out_percussive, samplerate=self.rate)

        lopass_filter = os.path.join(self.out_bass, 'output_bass.wav') #path of bass
        print('Saving Low pass filtered audio to: ', lopass_filter)
        sf.write(file=lopass_filter, data=out_bass, samplerate=self.rate)

        print('Save Complete Check folder to hear the results.' )
       
    def harm_perc_bass_plot_spec(self):
        P = self.get_outputs() 
        #bass = self.separate_bass()
        y_harm = librosa.stft(P[0], n_fft=2048, hop_length=512)
        y_perc = librosa.stft(P[1], n_fft=2048, hop_length=512)
        y_bass = librosa.stft(P[2], n_fft=2048, hop_length=512)

        P_harm = librosa.power_to_db(np.abs(y_harm)**2)
        P_perc = librosa.power_to_db(np.abs(y_perc)**2)
        P_bass = librosa.power_to_db(np.abs(y_bass)**2)

        plt.figure(figsize=(10, 6))

        plt.subplot(3, 1, 1)
        librosa.display.specshow(P_harm, y_axis='log')
        plt.colorbar(format="%+2.f")
        plt.title('Harmonic spectrogram')
        
        plt.subplot(3, 1, 2)
        librosa.display.specshow(P_perc, y_axis='log')
        plt.colorbar(format="%+2.f")
        plt.title('Percussive spectrogram')
        plt.tight_layout()

        plt.subplot(3, 1, 3)
        librosa.display.specshow(P_bass, y_axis='log', x_axis='time')
        plt.colorbar(format="%+2.f")
        plt.title('Bass spectrogram')
        plt.tight_layout()
        plt.show()
