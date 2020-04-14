import os
import pickle

import librosa
import numpy as np
import matplotlib.pyplot as plt

def pick_peaks(spectrogram, tau=16, kappa=32, hop_tau=4, hop_kappa=64):
    # create empty array to store peaks
    peaks = np.zeros_like(spectrogram)

    # calculate how many hops we will make along each axis
    n_freq_steps = int(np.floor((spectrogram.shape[0] - 2 * kappa) / hop_kappa))
    n_time_steps = int(np.floor((spectrogram.shape[1] - 2 * tau) / hop_tau))

    for n in range(n_time_steps):
        for k in range(n_freq_steps):
            # calculate the bounds of our window
            freq_lower = k * hop_kappa
            freq_upper = k * hop_kappa + 2 * kappa
            time_lower = n * hop_tau
            time_upper = n * hop_tau + 2 * tau

            # find the time and frequency indices of the peak in our window
            window = spectrogram[freq_lower:freq_upper, time_lower:time_upper]
            peak = np.unravel_index(np.argmax(window), window.shape)

            # store the peak in the main array
            peaks[freq_lower + peak[0], time_lower + peak[1]] = 1
    
    return peaks

def quantise_frequency(peaks, factor=32):
    # create an empty array to store the output
    n_bins = int(np.ceil(peaks.shape[0] / factor))
    quantised_peaks = np.zeros((n_bins, peaks.shape[1]))

    for k in range(n_bins):
        # extract the group of frequencies we will be merging
        freq_lower = k * factor
        freq_upper = (k + 1) * factor
        freq_group = peaks[freq_lower:freq_upper, :]

        # sum across the axis and keep only on/off values
        new_row = np.sum(freq_group, axis=0)
        new_row[new_row > 0] = 1

        # store row in output array
        quantised_peaks[k, :] = new_row
    
    return quantised_peaks


def create_inverted_lists(peaks):
    inverted_lists = []
    for row in peaks:
        inverted_list = np.nonzero(row)
        inverted_lists.append(inverted_list[0])
    return inverted_lists

def create_fingerprint(path_to_audio):
    # load audio
    x, sr = librosa.load(path_to_audio)

    # compute STFT
    X = np.abs(librosa.core.stft(x))

    # pick peaks
    peaks = pick_peaks(X)

    # quantise peaks in frequency
    quantised_peaks = quantise_frequency(peaks)

    return quantised_peaks


def fingerprintBuilder(path_to_db, path_to_fingerprints):
    for entry in os.scandir(path_to_db):
        if os.path.splitext(entry.name)[1] != ".wav":
            continue

        print("Creating fingerprint for %s..." % entry.name, end="\r")
        fingerprint = create_fingerprint(entry.path)
        # create inverted lists for each frequency bin
        inverted_lists = create_inverted_lists(fingerprint)

        print("Saving fingerprint for %s..." % entry.name, end="\r")
        output_file_name = os.path.splitext(entry.name)[0] + ".fingerprint"
        output_path = os.path.join(path_to_fingerprints, output_file_name)

        with open(output_path, "wb") as f:
            pickle.dump(inverted_lists, f)

        print("%s done." % entry.name, end="\n")