"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 2: Audio Identification

File: audio_identification.py
Description: Searches a fingerprint database for likely matches to a folder of
             query audio files.
"""
import os
import pickle
import time

import numpy as np

from fingerprint_builder import extract_spectral_peaks, create_pairwise_hashes
from print_status import print_status, enable_printing


def get_query_hashes(
        query_file,
        peak_picking_options={},
        pair_searching_options={}):
    """
    Given the path of a query audio file, return a list of pairwise spectral
    peak hashes present in the query.
    
    Arguments:
        query_file {str} -- Path to query audio
    
    Keyword Arguments:
        peak_picking_options {dict} -- Optional dict of peak picking options
                                       (default: {{}})
        pair_searching_options {dict} -- Optional dict of pair searching 
                                         options (default: {{}})
    """        

    query_fingerprint =\
        extract_spectral_peaks(query_file, peak_picking_options)
    query_hashes =\
        create_pairwise_hashes(query_fingerprint, **pair_searching_options)

    return query_hashes


def find_potentially_matching_docs(query_hashes, doc_hashes):
    """
    Given a list of hashes present in a query and a hash table linked hashes
    to document IDs and time offsets, return all potentially matching documents
    with a list of relevant hashes and the difference between their time
    offsets in the query and document.
    
    Arguments:
        query_hashes {list} -- List of hashes present in the query
        doc_hashes {dict} -- Hash table linking hashes to documents
    
    Returns:
        dict -- Dictionary linking document IDs to lists of relevant hashes and
                their offset time deltas.
    """    

    # initialise dict for docs sharing hashes with query
    query_docs = {}
    # iterate over query hashes
    for hash in query_hashes:
        # if we have seen this hash in our database
        if hash["hash"] in doc_hashes:
            # for every document associated with it
            for hash_match in doc_hashes[hash["hash"]]:
                # start a list if we don't have one already
                if hash_match["name"] not in query_docs:
                    query_docs[hash_match["name"]] = []
            
                # add the time difference between the query hash and the
                # document hash under the document's key in our dict of
                # potentially matching docs
                query_docs[hash_match["name"]].append(
                    hash_match["offset"] - hash["offset"])
    
    return query_docs

def compute_histogram_ranges(query_docs):
    """
    Given a dict associating docs to a list of hashes and their offset time
    deltas, compute a histogram for each and return a dict associating docs
    to the range of their histograms.
    
    Arguments:
        query_docs {dict} -- Dictionary linking document IDs to a list of their
                             hashes and time deltas
    
    Returns:
        dict -- Dictionary associating docs to their histogram ranges
    """    
    # initialise the dict which will contain the ranges of the histograms
    # of time offsets
    histogram_ranges = {}

    # iterate over each potentially matching doc
    for doc in query_docs:
        # skip any zero length docs
        if len(query_docs[doc]) == 0:
            continue

        # create a histogram — NB np.bincount is faster than np.histogram
        # but at the expense of being able to create any sized bin
        histogram = np.bincount(query_docs[doc] - np.min(query_docs[doc]))

        # store the min to max range of the histogram in our dict
        histogram_ranges[doc] = int(np.ptp(histogram))

    return histogram_ranges


def sort_flat_dict(dict_to_sort):
    """
    Given a dictionary of depth 1, return a list of keys sorted by values
    
    Arguments:
        dict_to_sort {dict} -- A dict of depth 1
    
    Returns:
        list -- A list of keys sorted by their values in dict_to_sort
    """    
    scores = [key for key in dict_to_sort]
    keys_by_scores = sorted(scores, key=lambda x: -dict_to_sort[x])
    return keys_by_scores


def doc_matches_query(doc_name, query_name):
    """
    Returns true if the doc name matches the ground truth in the query name,
    based on the filename formatting in the GTZAN dataset.
    
    Arguments:
        doc_name {str} -- The document file name
        query_name {str} -- The query file name
    
    Returns:
        boolean -- True if the document matches the query
    """    
    ground_truth = query_name.split("-")[0] + ".wav"
    return ground_truth == doc_name


def write_output_line(output_file, sorted_docs, query_name):
    if len(sorted_docs) > 0:
        output_line = "%s\t%s\n" % (
            query_name,
            "\t".join(sorted_docs[:min(3, len(sorted_docs))]))
    else:
        output_line = query_name
    output_file.write(output_line)


@enable_printing
def audioIdentification(
        path_to_queries,
        path_to_fingerprints,
        path_to_output,
        peak_picking_options={},
        pair_searching_options={}):
    """
    The main entry point for the audio identifying algorithm
    
    Arguments:
        path_to_queries {str} -- Path to query audio directory
        path_to_fingerprints {str} -- Path to fingerprint database file
        path_to_output {str} -- Path to output text file
    
    Keyword Arguments:
        peak_picking_options {dict} -- Optional dict of keyword args to peak
                                       picking algorithm (default: {{}})
        pair_searching_options {dict} -- Optional dict of keyword args to pair
                                         searching algorithm (default: {{}})
    
    Returns:
        [type] -- [description]
    """
    print_status("id_blank_status", {})
    print_status("id_loading_db", {"db_file": path_to_fingerprints})

    start_time = time.perf_counter()
    # open output file for writing
    output_file = open(path_to_output, "w")

    # load fingerprint database from disk
    with open(path_to_fingerprints, "rb") as f:
        fingerprints = pickle.load(f)

    # initialise counters
    n_queries = 0
    n_correct = 0

    # iterate over files in query directory
    for entry in os.scandir(path_to_queries):
        # skip any files that aren't WAVs
        if os.path.splitext(entry.name)[1] != ".wav":
            continue
        n_queries += 1

        # extract hashes from query (and time it + report status)
        print_status(
            "id_analysing_file",
            { "now_analysing": entry.name })
        hash_start_time = time.perf_counter()
        query_hashes = get_query_hashes(
            entry.path,
            peak_picking_options,
            pair_searching_options)
        hash_time = time.perf_counter() - hash_start_time

        print_status(
            "id_searching_db",
            { "now_analysing": entry.name })
        db_search_start_time = time.perf_counter()

        # find all docs sharing hashes with the query, and construct a dict of
        # their names and the time deltas between the hash time in the query
        # and the doc:
        query_docs =\
            find_potentially_matching_docs(query_hashes, fingerprints)

        # find the ranges of histograms of their time deltas:
        histogram_ranges = compute_histogram_ranges(query_docs)
        
        # sort the docs in order of their negative histogram ranges — i.e. best
        # match first
        sorted_docs = sort_flat_dict(histogram_ranges)

        # compare first result to ground truth and find out if we are correct
        correct = len(sorted_docs) > 0\
                and doc_matches_query(sorted_docs[0], entry.name)
        n_correct += 1 if correct else 0
        
        db_search_time = time.perf_counter() - db_search_start_time

        print_status(
            "id_finished_identifying",
            {
                "file_name": entry.name,
                "correctly_identified": "Yes" if correct else "No",
                "correct_so_far":
                    "%.1f%%" % (100 * float(n_correct) / n_queries),
                "guess_1": sorted_docs[0] if len(sorted_docs) >= 1 else "",
                "guess_2": sorted_docs[1] if len(sorted_docs) >= 2 else "",
                "guess_3": sorted_docs[2] if len(sorted_docs) >= 3 else "",
                "time_to_hashes": "%.3f" % hash_time,
                "time_to_db": "%.3f" % db_search_time,
                "total_time": "%.1f" % (time.perf_counter() - start_time)
            }
        )

        write_output_line(output_file, sorted_docs, entry.name)

    output_file.close()

    return float(n_correct) / n_queries
