import os
import pickle
import time

import numpy as np

from fingerprint_builder import create_fingerprint, create_pairwise_hashes
from print_status import print_status, enable_printing


def get_query_hashes(
        query_file,
        peak_picking_options={},
        pair_searching_options={}):

    query_fingerprint =\
        create_fingerprint(query_file, peak_picking_options)
    query_hashes =\
        create_pairwise_hashes(query_fingerprint, **pair_searching_options)

    return query_hashes


@enable_printing
def audioIdentification(
        path_to_queries,
        path_to_fingerprints,
        path_to_output,
        peak_picking_options={},
        pair_searching_options={}):

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

        print_status(
            "id_analysing_file",
            { "now_analysing": entry.name })

        hash_start_time = time.perf_counter()
        # extract hashes from query
        query_hashes = get_query_hashes(
            entry.path,
            peak_picking_options,
            pair_searching_options)
        hash_time = time.perf_counter() - hash_start_time

        print_status(
            "id_searching_db",
            { "now_analysing": entry.name })

        db_search_start_time = time.perf_counter()
        # initialise dict for docs sharing hashes with query
        query_docs = {}

        # iterate over query hashes
        for hash in query_hashes:
            # if we have seen this hash in our database
            if hash["hash"] in fingerprints:
                # for every document associated with it
                for hash_match in fingerprints[hash["hash"]]:
                    # start a list if we don't have one already
                    if hash_match["name"] not in query_docs:
                        query_docs[hash_match["name"]] = []
                
                    # add the time difference between the query hash and the
                    # document hash under the document's key in our dict of
                    # potentially matching docs
                    query_docs[hash_match["name"]].append(
                        hash_match["offset"] - hash["offset"])

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
        
        # sort the histograms by ranges (big range suggests a match, as we have
        # many hashes offset by the same amount)
        scores = [key for key in histogram_ranges]
        sorted_scores = sorted(scores, key=lambda x: -histogram_ranges[x])

        # compare to ground truth and find out if we are correct
        ground_truth = entry.name.split("-")[0] + ".wav"
        if len(sorted_scores) > 0 and ground_truth == sorted_scores[0]:
            n_correct += 1
            correct = True
        else:
            correct = False
        
        db_search_time = time.perf_counter() - db_search_start_time

        print_status(
            "id_finished_identifying",
            {
                "file_name": entry.name,
                "correctly_identified": "Yes" if correct else "No",
                "correct_so_far":
                    "%.1f%%" % (100 * float(n_correct) / n_queries),
                "guess_1": sorted_scores[0] if len(sorted_scores) >= 1 else "",
                "guess_2": sorted_scores[1] if len(sorted_scores) >= 2 else "",
                "guess_3": sorted_scores[2] if len(sorted_scores) >= 3 else "",
                "time_to_hashes": "%.3f" % hash_time,
                "time_to_db": "%.3f" % db_search_time,
                "total_time": "%.1f" % (time.perf_counter() - start_time)
            }
        )

        if len(sorted_scores) > 0:
            output_line =\
                "%s %s\n" % (entry.name, ", ".join(sorted_scores[:3]))
            output_file.write(output_line)
            
    output_file.close()

    return float(n_correct) / n_queries
