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
outvocals = outfiledir + '\out_vocals'

my_objects = []
length = len(mySongs)

for i in range(length):
    my_objects.append(Separate(os.path.join(filedir, mySongs[i]), mySongs[i], AudioSegment.from_file( os.path.join(filedir, mySongs[i])).frame_rate, outharm, outperc, outbass, outvocals))

# for obj in my_objects:
#      print(obj)

def main():
    '''
    Run main 
    -------------Separate song of choice in 4 stems------------------
    --------------1. Harmonic Part-------------------------------
    --------------2. Percussive Part-----------------------------
    --------------3. Bass Part-----------------------------------
    --------------4. Acapella Part-------------------------------

    How to Run
        -Select input song by the List of input Songs (song_input directory), songs start from 0, example my_object[0] refers to first song
        -First off the program plots the spectrogram of input song of choice
        -Then it plots the spectrogramm of all 4 stems 
        -Finally the program saves the stems into our song_output directory in their respective folder (harmonic, percussive, bass, vocals)

        *Each time you run it with another song the outputs are overwritting*

        ------Average time to Separate Harmonic and Percussive is: 10 secs------
        ------Average time to Separate Vocals is: 46 secs-----------------------
        ------Average time to Save results is: 4 secs---------------------------
    
    '''
    # my_objects[0].plot_input_spectrogram()
    # my_objects[0].harm_perc_bass_plot_spec()
    my_objects[1].save_output()

if __name__ == "__main__":
    main()