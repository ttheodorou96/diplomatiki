# README
Decomposition tool: 

Ο χρήστης εισάγει στο directory "song_input" όσα αρχεία .mp3/wav επιθυμεί, το σύστημα αρχικά εμφανίζει το spectrogram του αρχείου
(ο χρήστης επιλέγει ποιό κομμάτι απο αυτά που υπάρχουν στο directory θα χειριστεί) και στην συνέχεια χωρίζει το μουσικό κομμάτι που έχει εισαχθεί απο τον χρήστη σε stems/components. Τα components αυτά είναι:
- Harmonic Part
- Percussive Part 
- Bass Part

Στην συνέχεια εμφανίζει το spectrogram των τριών αυτών components 
και τέλος η κονσόλα ενημερώνει τον χρήστη τα directories που έχουν αποθηκευτεί τα stems μας (/output_song/out_harm, /output_song/out_perc, /output_song/out_bass) σε μορφή .wav.

Το παραπάνω σύστημα έχει αναπτυχθεί σε γλώσσα python και σε πρόγραμμα επεξεργασίας Visual Studio Code. Είναι βασισμένο σε αντικειμενοστραφή προγραμματισμό με χρήση συναρτήσεων και κλάσεων. 

Για εκτέλεση τρέχεις το __main__.py αρχείο

 
Python 3.6.12 :: Anaconda, Inc.

conda 4.8.3



bibliography

[1] Harmonic-percussive source separation
librosa-gallery

https://librosa.org/librosa_gallery/auto_examples/plot_hprss.html

[2] Vocal separation
librosa-gallery

https://librosa.org/librosa_gallery/auto_examples/plot_vocal_separation.html

[3] MUSIC/VOICE SEPARATION USING THE SIMILARITY MATRIX
Zafar RAFII, Bryan PARDO
https://users.cs.northwestern.edu/~zra446/doc/Rafii-Pardo%20-%20Music-Voice%20Separation%20using%20the%20Similarity%20Matrix%20-%20ISMIR%202012.pdf

[4] Vocal Separation Using Nearest Neighbours and Median Filtering
Derry Fitzgerald 
https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1086&context=argcon

[5] Separation of Instrument Sounds using
Non-negative Matrix Factorization with Spectral
Envelope Constraints
https://arxiv.org/pdf/1801.04081.pdf

[6] librosa: Audio and Music Signal Analysis in Python

https://brianmcfee.net/papers/scipy2015_librosa.pdf

[7] Librosa Examples 
https://github.com/librosa/librosa/tree/main/examples
