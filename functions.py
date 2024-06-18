import numpy as np
from scipy import fft, signal
from scipy.io.wavfile import read
from scipy.fft import fftfreq
import numpy as np
import scipy.io.wavfile as wav
import os
from pydub import AudioSegment
import librosa

def create_constellation(audio, Fs):
    # Parameters
    window_length_seconds = 0.5
    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 15
    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples
    song_input = np.pad(audio, (0, amount_to_pad))
    # Perform a short time fourier transform
    frequencies, times, stft = signal.stft(
        song_input, Fs, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True
    )
    constellation_map = []
    for time_idx, window in enumerate(stft.T):
        spectrum = abs(window)
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        n_peaks = min(num_peaks, len(peaks))
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])
    return constellation_map

def create_hashes(constellation_map, song_id=None):
    hashes = {}
    # Use this for binning - 23_000 is slighlty higher than the maximum
    # frequency that can be stored in the .wav files, 22.05 kHz
    upper_frequency = 23_000 
    frequency_bits = 10
    # Iterate the constellation
    for idx, (time, freq) in enumerate(constellation_map):
        # Iterate the next 100 pairs to produce the combinatorial hashes
        # When we produced the constellation before, it was sorted by time already
        # So this finds the next n points in time (though they might occur at the same time)
        for other_time, other_freq in constellation_map[idx : idx + 100]: 
            diff = other_time - time
            # If the time difference between the pairs is too small or large
            # ignore this set of pairs
            if diff <= 1 or diff > 10:
                continue
            # Place the frequencies (in Hz) into a 1024 bins
            freq_binned = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_binned = other_freq / upper_frequency * (2 ** frequency_bits)
            # Produce a 32 bit hash
            # Use bit shifting to move the bits to the correct location
            hash = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash] = (time, song_id)
    return hashes

def score_hashes_against_database(hashes, database):
    matches_per_song = {}
    for hash, (sample_time, _) in hashes.items():
        if hash in database:
            matching_occurences = database[hash]
            for source_time, song_index in matching_occurences:
                if song_index not in matches_per_song:
                    matches_per_song[song_index] = []
                matches_per_song[song_index].append((hash, sample_time, source_time))
            

    scores = {}
    for song_index, matches in matches_per_song.items():
        song_scores_by_offset = {}
        for hash, sample_time, source_time in matches:
            delta = source_time - sample_time
            if delta not in song_scores_by_offset:
                song_scores_by_offset[delta] = 0
            song_scores_by_offset[delta] += 1
        max = (0, 0)
        for offset, score in song_scores_by_offset.items():
            if score > max[1]:
                max = (offset, score)
        
        scores[song_index] = max
    # Sort the scores for the user
    scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True)) 
    
    return scores

def print_top_five(file_name, database, song_name_index):
    # Load a short recording with some background noise
    Fs, audio_input = read(file_name)
    # Create the constellation and hashes
    constellation = create_constellation(audio_input, Fs)
    hashes = create_hashes(constellation, None)
    scores = score_hashes_against_database(hashes, database)[:5]
    for song_id, score in scores:
        print(f"{song_name_index[song_id]}: Score of {score[1]} at {score[0]}")

def print_top_one(file_name, database, song_name_index):
    # Load a short recording with some background noise
    Fs, audio_input = read(file_name)
    # Create the constellation and hashes
    constellation = create_constellation(audio_input, Fs)
    hashes = create_hashes(constellation, None)
    scores = score_hashes_against_database(hashes, database)[:1]
    return scores

# WHITE NOISE
def add_white_noise(signal, noise_level):
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_level * noise
    return augmented_signal

def create_white_noise_files(original_song, folder_name):
    for noise in np.arange(0, 3000, 75):
        rate, data = wav.read(original_song)
        noisy_data = add_white_noise(data, noise)
        file_location = f'{folder_name}/white_noise_{noise}.wav'
        wav.write(file_location, rate, noisy_data.astype(np.int16))

def scores_for_different_noise(directory, database, song_name_index):
    list_of_right_songs = {}
    scores = []
    for audio in os.listdir(directory):
        score = print_top_one(f'{directory}{audio}', database, song_name_index)
        list_of_right_songs[audio] = song_name_index[score[0][0]]
        scores.append(score[0][1][1])
    return list_of_right_songs, scores

# CLIPPING
def add_clipping_distortion(signal, threshold = 0.7):
    clipped_signal = np.clip(signal, -threshold, threshold)
    return clipped_signal

def create_clipped_files(original_song, folder_name):
    for clipping in np.arange(250, 10250, 250):
        rate, data = wav.read(original_song)
        clipped_data = add_clipping_distortion(data, clipping)
        file_location = f'{folder_name}/clipping_{clipping}.wav'
        wav.write(file_location, rate, clipped_data.astype(np.int16))

# PITCH SHIFTING
def time_stretch(signal, stretch_factor):
    stretched_signal = librosa.effects.time_stretch(y = signal.astype(float), rate = stretch_factor)
    return stretched_signal

def pitc_shift(signal, rate, n_steps):
    shifted_signal = librosa.effects.pitch_shift(y = signal.astype(float), sr = rate, n_steps = n_steps)
    return shifted_signal

def create_pitched_files(original_song, folder_name):
    for pitc in np.arange(0, 10, 0.25):
        rate, data = wav.read(original_song)
        clipped_data = pitc_shift(data, rate, pitc)
        file_location = f'{folder_name}/pitched_{pitc}.wav'
        wav.write(file_location, rate, clipped_data.astype(np.int16))

# SHORTER FILES
def shorten_file(seconds, directory, new_directory):
    milliseconds = seconds * 1000    
    os.makedirs(new_directory, exist_ok = True)    
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            audio = AudioSegment.from_wav(os.path.join(directory, file))
            short_audio = audio[:milliseconds]  # pydub calculates in milliseconds
            short_audio.export(os.path.join(new_directory, file), format='wav')

def scores_different_lenght(directory, database, song_name_index):
    conteggio = 0
    for file in os.listdir(directory):
        score = print_top_one(f'{directory}/{file}', database, song_name_index)
        song_name_guessed = song_name_index[score[0][0]].replace('converted/', '')
        if song_name_guessed != file:
            pass
        else:
            conteggio += 1
    return conteggio

# REMOVE FILES
def remove_created_audio_files_start(directory):
    print('----------------------')
    for dir in os.listdir('pitch'):
        for audio in os.listdir(directory + '/' + dir):
            os.remove(directory + '/' + dir + '/' + audio)
        print('deleted files in directory ' + directory + '/' + dir)
    print('----------------------')

def remove_created_audio_files_end(directory):
    print('----------------------')
    for dir in os.listdir('pitch'):
        for audio in os.listdir(directory + '/' + dir):
            os.remove(directory + '/' + dir + '/' + audio)
        file_path = directory + '/' + dir + '/.gitkeep'
        fd = os.open(file_path, os.O_CREAT | os.O_WRONLY)
        os.close(fd)
        print('deleted files in directory ' + directory + '/' + dir)
    print('----------------------')

def remove_short_recordings_end(directory):
    for dir in os.listdir(directory):
        if dir == 'original':
            continue
        else:
            for file in os.listdir(directory + '/' + dir):
                os.remove(directory + '/' + dir + '/' + file)
            print('Removed files in ' + directory + '/' + dir)
            file_path = directory + '/' + dir + '/.gitkeep'
            fd = os.open(file_path, os.O_CREAT | os.O_WRONLY)
            os.close(fd)

def remove_short_recordings_start(directory):
    for dir in os.listdir(directory):
        if dir == 'original':
            continue
        else:
            for file in os.listdir(directory + '/' + dir):
                os.remove(directory + '/' + dir + '/' + file)
            print('Removed files in ' + directory + '/' + dir)