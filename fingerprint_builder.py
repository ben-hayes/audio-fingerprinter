import curses
import os
import pickle
import time

import librosa
import numpy as np

from print_status import print_status, enable_printing


def pick_peaks(spectrogram, tau=21, kappa=59, hop_tau=11, hop_kappa=74):
    """
    Given a spectrogram, finds peaks within window specified by parameters.
    Window shape will be (2 * kappa + 1, 2 * tau + 1)

    Default parameters selected by random search.
    
    Arguments:
        spectrogram {NumPy Array} -- Time-frequency magnitude representation of
                                     signal
    
    Keyword Arguments:
        tau {int} --  Window size in time direction (default: {21})
        kappa {int} -- Window size in frequency direction (default: {59})
        hop_tau {int} -- Hop size in time direction (default: {11})
        hop_kappa {int} -- Hop size in frequency direction (default: {74})
    
    Returns:
        NumPy Array -- A sparse NumPy array of same shape as the input. Peaks
                       will be signified by ones and non-peaks by zeros.
    """    
    # create empty array to store peaks
    peaks = np.zeros_like(spectrogram)

    # calculate how many hops we will make along each axis
    n_freq_steps =\
        int(np.floor((spectrogram.shape[0] - 2 * kappa) / hop_kappa))
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


def find_peak_pairs(
        peaks,
        target_time_offset=3,
        target_time_width=196,
        target_freq_height=220):
    """
    Given a sparse array of spectral peaks, find all peak pairs according to
    a given set of window parameters.
    
    Arguments:
        peaks {NumPy Array} -- Sparse array of spectral peaks
    
    Keyword Arguments:
        target_time_offset {int} -- Offset of window from peak (default: {3})
        target_time_width {int} -- Width of window in time (default: {196})
        target_freq_height {int} -- Height of window in frequency
                                    (default: {220})
    """        

    # create an array of shape (N, 2) of the co-ordinates of all N peaks in
    # our sparse peak array
    points = np.dstack(np.nonzero(peaks)).squeeze()

    for point in points:
        # define the bounds of our target zone
        freq_lo = max(point[0] - target_freq_height, 0)
        freq_hi = min(point[0] + target_freq_height, peaks.shape[0])
        time_lo = point[1] + target_time_offset
        time_hi = time_lo + target_time_width

        # if we've passed the end of the array then skip this iteration
        if time_lo > peaks.shape[1]:
            continue

        # pull our target zone out of the sparse peak array
        target_zone = peaks[freq_lo:freq_hi, time_lo:time_hi]

        # again create another array of point co-ordinates, this time just in
        # our target zone
        target_points = np.dstack(np.nonzero(target_zone)).reshape(-1, 2)
        for target_point in target_points:
            # create our hashable tuple: (k_1, k_2, n_2 - n_1)
            peak_pair = (
                point[0],
                freq_lo + target_point[0],
                (time_lo + target_point[1]) - point[1])
            
            # using yield rather than return, we can call this function as a
            # generator which means we can write a pretty dict comprehension
            # and take advantage of some of python's (modest) optimisations:
            yield {
                "peak_pair": peak_pair,
                "offset": point[1]
            }

def create_pairwise_hashes(
        peaks,
        target_time_offset=3,
        target_time_width=196,
        target_freq_height=220):
    """
    Given a sparse array of spectral peaks, create a list of peak pair hashes
    and their corresponding time offsets.
    
    Arguments:
        peaks {NumPy Array} -- Sparse array of spectral peaks
    
    Keyword Arguments:
        target_time_offset {int} -- Offset of window from peak (default: {3})
        target_time_width {int} -- Width of window in time (default: {196})
        target_freq_height {int} -- Height of window in frequency
                                    (default: {220})
    """        

    # find all our peak pairs according to the criteria passed as arguments,
    # and construct a list of them along with their offset times. Note that as
    # we will be storing these in a Python dict (the fastest implementation of 
    # a hash table available to us in Python) and tuples are immutable and
    # therefore hashable, we don't need to explicitly calculate a hash value
    # and can instead directly use them as keys
    return [{
            "hash": pair["peak_pair"],
            "offset": int(pair["offset"])
        } for pair in find_peak_pairs(
            peaks,
            target_time_offset,
            target_time_width,
            target_freq_height
        )]


def create_fingerprint(path_to_audio, peak_picking_options={}):
    """
    Given an audio file, create a fingerprint (sparse array of spectral peaks)
    
    Arguments:
        path_to_audio {str} -- Path on disk to audio file
    
    Keyword Arguments:
        peak_picking_options {dict} -- Optional dict of keyword args to peak
                                       picking alogrithm. Useful for performing
                                       searches across parameter space for
                                       optimal combinations. (default: {{}})
    
    Returns:
        [type] -- [description]
    """    
    # load audio
    x, _ = librosa.load(path_to_audio)

    # compute STFT
    X = np.abs(librosa.core.stft(x))

    # pick peaks
    peaks = pick_peaks(X, **peak_picking_options)

    return peaks


@enable_printing
def fingerprintBuilder(
        path_to_db,
        path_to_fingerprints,
        peak_picking_options={},      
        pair_searching_options={}):   
    """
    The main entry point for our fingerprint builder application.
    
    Arguments:
        screen {curses.window} -- Reference to console window auto-created by
                                  curses.wrapper call
        path_to_db {str} -- Path to folder containing audio files
        path_to_fingerprints {str} -- Path to desired output file
    
    Keyword Arguments:
        peak_picking_options {dict} -- Optional dict of keyword args to peak
                                       picking algorithm (default: {{}})
        pair_searching_options {dict} -- Optional dict of keyword args to pair
                                         searching algorithm (default: {{}})
    """        

    # setup curses library
    curses.use_default_colors()
    # initialise timer
    start_time = time.perf_counter()

    # initialise our fingerprints dict
    fingerprints = {}

    # last count of fingerprints is useful for tracking how many new hashes
    # each file contributes
    last_fingerprints_length = 0

    for entry in os.scandir(path_to_db):
        # skip over non-wav files
        if os.path.splitext(entry.name)[1] != ".wav":
            continue

        print_status(
            "fp_analysing_fingerprint", {"now_analysing": entry.name}
        )

        # start timing hash creation
        hash_start_time = time.perf_counter()

        # pick out spectral peaks
        fingerprint = create_fingerprint(entry.path, peak_picking_options)
        # compute hashes
        hashes = create_pairwise_hashes(fingerprint, **pair_searching_options)

        for hash in hashes:
            # if we haven't seen this hash before - computable in O(1)
            if hash["hash"] not in fingerprints:
                # create an empty list at this hash's address
                fingerprints[hash["hash"]] = []

            # append the appropriate file name and time offset under this hash
            fingerprints[hash["hash"]].append({
                "name": entry.name,
                "offset": hash["offset"]
            })

        # find the current time to calculate performance
        time_now = time.perf_counter()
        print_status(
            "fp_fingerprint_created",
            {
                "file_name": entry.name,
                "num_hashes": len(hashes),
                "num_new_hashes":
                    len(fingerprints) - last_fingerprints_length,
                "total_hashes": len(fingerprints),
                "time_to_create": "%.3f" % (time_now - hash_start_time),
                "total_time": "%.3f" % (time_now - start_time)
            })
        last_fingerprints_length = len(fingerprints)

    print_status(
        "fp_writing_db",
        { "db_file": path_to_fingerprints }
    )

    # write the database to disk
    with open(path_to_fingerprints, "wb") as f:
        # using HIGHEST_PROTOCOL allows pickle to read/write faster and deal
        # with bigger files
        pickle.dump(fingerprints, f, pickle.HIGHEST_PROTOCOL)
