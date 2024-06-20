# Shazam Fingerprint Algorithm from scratch
Developing a simple prototype of Shazam in Python.

### Notebook structure
- fingerprint_algorithm.ipynb : notebook that contains the explanation and implementation of the algorithm to a theoretical case. 
- fpal_test : nootebook testing of the algorithm in a real situation, implementation of recorded audio files for fingerpirnt recognition.
- integrazione.ipynb : notebook containing some tests to look further into the robustness of the algorithm to white noise, clipping distortion and pitch shifting

### Folders and resorces
- genres_original : folder containing more than 1000 songs in .wav format. Unlabeled, they are solely and exclusively for testing the algorithm.
- converted_memo : folder that contains .wav audio recordings (contain presence of noise).  Useful for testing the algorithm in non-ideal situations
- clipping, pitch, white_noise : empty folders needed for storing temporary audio files in integrazione.ipynb

### Sources and requirements
- requirements.py : easy install all the items usefull for this project
- functions.py : some functions needed for the notebooks, particularly integrazione.ipynb
- references.md : bibliography

### Costellation map

![image5](https://github.com/exdsgift/FrequencyFingerprint-Algorithm/blob/main/images/6.jpeg)

### Test with some different noise

![image1](https://github.com/exdsgift/FrequencyFingerprint-Algorithm/blob/main/images/1.png)
![image1](https://github.com/exdsgift/FrequencyFingerprint-Algorithm/blob/main/images/2.png)
![image1](https://github.com/exdsgift/FrequencyFingerprint-Algorithm/blob/main/images/3.png)

### Snippet with different lenght
![image4](https://github.com/exdsgift/FrequencyFingerprint-Algorithm/blob/main/images/4.png)

