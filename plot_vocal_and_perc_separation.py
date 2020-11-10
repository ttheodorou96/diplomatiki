# Standard imports
from __future__ import print_function
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# Load an example with vocals.
y, sr = lb.load('/Users/alkis/Desktop/Python/Game Over-S.P.E.C.T.R.E. MASTER.wav', duration=120)


# And compute the spectrogram magnitude and phase
S_full, phase = lb.magphase(librosa.stft(y))
#######################################
# Plot a 5-second slice of the spectrum
idx = slice(*lb.time_to_frames([30, 35], sr=sr))
plt.figure(figsize=(12, 4))
lb.display.specshow(lb.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()

###########################################################

S_filter = lb.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)


##############################################
margin_i, margin_v = 2, 10
power = 2

mask_i = lb.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = lb.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

S_foreground = mask_v * S_full
S_background = mask_i * S_full


##########################################

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
lb.display.specshow(lb.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
lb.display.specshow(lb.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
lb.display.specshow(lb.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()

####################################################################
# perc separation
x, sr = lb.load('/Users/alkis/Desktop/Python/Game Over-S.P.E.C.T.R.E. MASTER.wav', offset=40, duration=10)
# Compute the short-time Fourier transform of x
D = librosa.stft(x)
# Decompose D into harmonic and percussive components * D = Dharmonic + Dpercussive *
D_harmonic, D_percussive = librosa.decompose.hpss(D)
# We can plot the two components along with the original spectrogram

# Pre-compute a global reference power from the input spectrum
rp = np.max(np.abs(D))

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=rp), y_axis='log')
plt.colorbar()
plt.title('Full spectrogram')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
plt.colorbar()
plt.title('Harmonic spectrogram')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Percussive spectrogram')
plt.tight_layout()

D_harmonic16, D_percussive16 = librosa.decompose.hpss(D, margin=16)


#############################################################################
# In the plots below, note that vibrato has been suppressed from the harmonic
# components, and vocals have been suppressed in the percussive components.
plt.figure(figsize=(10, 5))

plt.subplot(5, 2, 5)
librosa.display.specshow(librosa.amplitude_to_db(D_harmonic16, ref=rp), y_axis='log')
plt.yticks([])
plt.ylabel('margin=16')

plt.subplot(5, 2, 6)
librosa.display.specshow(librosa.amplitude_to_db(D_percussive16, ref=rp), y_axis='log')
plt.yticks([]), plt.ylabel('')

plt.tight_layout()
plt.show()
