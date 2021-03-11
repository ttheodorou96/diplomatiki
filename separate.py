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
        super().__init__(path, filename, rate, out_harm, out_perc, out_bass, out_vocals, input_song)

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

    def vocal_separation_demo(self):
        print('Separating Vocals...')
        S_full, phase = librosa.magphase(librosa.stft(self.input_song))
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine',
                                            width=int(librosa.time_to_frames(2, sr=self.rate)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2

        # mask_i = librosa.util.softmask(S_filter,
        #                             margin_i * (S_full - S_filter),
        #                             power=power)
        mask_v = librosa.util.softmask(S_full - S_filter,
                                    margin_v * S_filter,
                                    power=power)

        S_vocals = mask_v * S_full
        # S_background = mask_i * S_full

        return S_vocals

    def invert_stft(self):
        y = self.hpss_demo()
        y_vocals = self.vocal_separation_demo()
        y_harmonic = y[0]
        y_percussive = y[1]
        print('Separation Complete!\nApply Inverted STFT, ready to save results in wav form... ')
        out_harmonic = librosa.core.istft(y_harmonic) #type float32 can be used in sf.write.
        out_percussive = librosa.core.istft(y_percussive)
        out_vocals = librosa.core.istft(y_vocals)

        # --------test Types of numpy arrays---------#
        # print(out_harmonic.shape)
        # print(out_harmonic.dtype)
        return out_harmonic, out_percussive, out_vocals

    def butter_lowpass(self,lowcut, fs, order=5):
        """
        Design lowpass filter, returns the filter coefficients.

        :Args:
            - lowcut (float) : the cutoff frequency of the filter.
            - fs     (float) : the sampling rate.
            - order    (int) : order of the filter, by default defined to 5.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self,data, lowcut, fs, order=5):
        """
        Design lowpass filter, gets the filter coefficients and returns a numpy array of the filtered audio file.

        :Args:
            - data (ndarray) : the audio file stored in a numpy array. 
            - lowcut (float) : the cutoff frequency of the filter.
            - fs     (float) : the sampling rate.
            - order    (int) : order of the filter, by default defined to 5.
        """
        b, a = self.butter_lowpass(lowcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_highpass(self, highcut, fs, order=5):
        """
        Design highpass filter, returns the filter coefficients.

        :Args:
            - highcut (float) : the cutoff frequency of the filter.
            - fs     (float) : the sampling rate.
            - order    (int) : order of the filter, by default defined to 5.
        """
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, high, btype="highpass")
        return b, a

    def butter_highpass_filter(self, data, highcut, fs, order=5):
        """
        Design highpass filter, gets the filter coefficients and returns a numpy array of the filtered audio file.

        :Args:
            - data (ndarray) : the audio file stored in a numpy array. 
            - highcut (float) : the cutoff frequency of the filter.
            - fs     (float) : the sampling rate.
            - order    (int) : order of the filter, by default defined to 5.
        """
        b, a = self.butter_highpass(highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def get_outputs(self):

        # Filter requirements.
        order = 6
        fs = self.rate      # sample rate, Hz
        lowcut = 250  # desired cutoff frequency of the low-pass filter, Hz, float
        highcut = 500 # desired cutoff frequency of the high-pass filter, Hz, float
        upper = 2000 # desired upper cutoff frequency of the low-pass filter, Hz, float. Helps attenuate overtones 
        # Get the filter coefficients for highpass.
        b, a = self.butter_highpass(highcut, fs, order)
        # Get the filter coefficients for lowpass.
        b, a = self.butter_lowpass(lowcut, fs, order)
        output = self.invert_stft()

        out_harmonic = output[0]
        out_percussive = output[1]
        out_vocals = output[2]
        out_midrange = self.butter_highpass_filter(out_harmonic, highcut, fs, order)
        out_bass = self.butter_lowpass_filter(out_harmonic, lowcut, fs, order)
        out_midrange = self.butter_lowpass_filter(out_midrange, upper, fs, order)

        return out_percussive, out_bass, out_midrange ,out_vocals

    def save_output(self):
        out = self.get_outputs()
        
        out_percussive = out[0]
        out_bass = out[1]
        out_harmonic = out[2]
        out_vocals = out[3]
        
        # Save the results
        output_harmonic = os.path.join(self.out_harm, 'output_harmonic.wav') #path of out_harm
        print('Saving Harmonic audio file to: ', output_harmonic)
        sf.write(file=output_harmonic, data=out_harmonic, samplerate=self.rate)
        
        output_percussive = os.path.join(self.out_perc, 'output_percussive.wav') #path of perc
        print('Saving Percussive audio file to: ', output_percussive)
        sf.write(file=output_percussive, data=out_percussive, samplerate=self.rate)

        lopass_filter = os.path.join(self.out_bass, 'output_bass.wav') #path of bass
        print('Saving Bass audio file to: ', lopass_filter)
        sf.write(file=lopass_filter, data=out_bass, samplerate=self.rate)

        output_vocals = os.path.join(self.out_vocals, 'output_vocals.wav') #path of vocals
        print('Saving Acapella audio file to: ', output_vocals)
        sf.write(file=output_vocals, data=out_vocals, samplerate=self.rate)

        # output_harmonic = os.path.join(self.out_harm, 'output_midrange.wav') #path of out_harm
        # print('Saving harmonic audio to: ', output_harmonic)
        # sf.write(file=output_harmonic, data=out_midrange, samplerate=self.rate)

        print('Save Complete Check folder to hear the results.' )
       
    def harm_perc_bass_plot_spec(self):
        P = self.get_outputs() 
        
        y_perc = librosa.stft(P[0], n_fft=2048, hop_length=512)
        y_bass = librosa.stft(P[1], n_fft=2048, hop_length=512)
        y_harm = librosa.stft(P[2], n_fft=2048, hop_length=512)
        y_vocals = librosa.stft(P[3], n_fft=2048, hop_length=512)

        P_harm = librosa.power_to_db(np.abs(y_harm)**2)
        P_perc = librosa.power_to_db(np.abs(y_perc)**2)
        P_bass = librosa.power_to_db(np.abs(y_bass)**2)
        P_vocals = librosa.power_to_db(np.abs(y_vocals)**2)

        plt.figure(figsize=(12, 8))

        plt.subplot(4, 1, 1)
        librosa.display.specshow(P_harm, y_axis='log')
        plt.colorbar(format="%+2.f")
        plt.title('Harmonic spectrogram without bass')
        
        plt.subplot(4, 1, 2)
        librosa.display.specshow(P_perc, y_axis='log')
        plt.colorbar(format="%+2.f")
        plt.title('Percussive spectrogram')
        plt.tight_layout()

        plt.subplot(4, 1, 3)
        librosa.display.specshow(P_bass, y_axis='log')
        plt.colorbar(format="%+2.f")
        plt.title('Bass spectrogram')
        plt.tight_layout()

        plt.subplot(4, 1, 4)
        librosa.display.specshow(P_vocals, y_axis='log', x_axis='time')
        plt.colorbar(format="%+2.f")
        plt.title('Vocals spectrogram')
        plt.tight_layout()
        plt.show()
