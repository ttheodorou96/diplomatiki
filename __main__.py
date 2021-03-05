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
'''
    imported files
'''
import audio
from effects import Effects
from separate import Separate

'''
    Global Variables
'''

#get path of songs directory
cwd = os.getcwd()
filedir = cwd + '\song_input'
#create an array with the songs of the directory
mySongs = os.listdir(filedir) 
#get path of output songs directory
outfiledir = cwd + '\song_output'
outharm = outfiledir + '\out_harm'
outperc = outfiledir + '\out_perc'
outbass = outfiledir + '\out_bass'

my_objects = []
length = len(mySongs)

for i in range(length):
    my_objects.append(Separate(os.path.join(filedir, mySongs[i]), mySongs[i], AudioSegment.from_file( os.path.join(filedir, mySongs[i])).frame_rate, outharm, outperc, outbass))

# for obj in my_objects:
#      print(obj)

def main():
    '''
    Run main 
    -------------Separate first song in 3 stems------------------
    --------------1. Harmonic Part-------------------------------
    --------------2. Percussive Part-----------------------------
    --------------3. Bass Part-----------------------------------

    For better results in bass isolation: 
        -bass frequency spectrum range is 60-250Hz
        -Every song varies in this range of spectrum 
        -Run harm_perc_bass_plot_spec() to see range of Bass
        -Use different Supremum for each song by changing cutoff variable in separate.py
        -Later we can model this with an ML algorithm

    '''
    #my_objects[0].plot_input_spectrogram()
    #my_objects[1].harm_perc_bass_plot_spec()
    my_objects[1].save_output()

if __name__ == "__main__":
    main()