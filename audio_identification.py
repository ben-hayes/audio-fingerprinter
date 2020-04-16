import curses
import os
import pickle
import time

import numpy as np

from fingerprint_builder import create_fingerprint, create_pairwise_hashes


# store status printouts in a handy dict to save them clogging up main function
statuses = {
    "blank_status": {
        "text": """
====================================================================









--------------------------------------------------------------------

====================================================================
            """,
        "y": 0,
        "x": 0
    },
    "finished_identifying": {
        "text": """
====================================================================
Finished identifying query: {file_name}
Correctly Identified?       {correctly_identified}
Correct so far:             {correct_so_far}
Guess #1:                   {guess_1}
Guess #2:                   {guess_2}
Guess #3:                   {guess_3}
Time to extract hashes:     {time_to_hashes} seconds
Time to look up in DB:      {time_to_db} seconds
Time elapsed so far:        {total_time} seconds
--------------------------------------------------------------------

====================================================================
            """,
        "y": 0,
        "x": 0
    },
    "analysing_file": {
        "text": "Now identifying:            {now_analysing}",
        "y": 12,
        "x": 0
    },
    "searching_db": {
        "text": "Searching DB for matches to {now_analysing}...",
        "y": 12,
        "x": 0
    },
    "loading_db": {
        "text": "Loading fingerprint database {db_file} from disk...",
        "y": 12,
        "x": 0
    }
}

def print_status(screen, status, status_args):
    """
    Helper function for printing the status to the screen.
    
    Arguments:
        screen {curses.window} -- Reference to a curses window object
        status {str} -- Key in statuses dictionary of desired status
        status_args {dict} -- Dict of keyword arguments to format the status
                              string.
    """    
    # using curses in lieu of print to allow for multiline overwrites
    screen.addstr(
        statuses[status]["y"],
        statuses[status]["x"],
        statuses[status]["text"].format(**status_args)
    )
    screen.refresh()


def get_query_hashes(
        query_file,
        peak_picking_options={},
        pair_searching_options={}):

    query_fingerprint =\
        create_fingerprint(query_file, peak_picking_options)
    query_hashes =\
        create_pairwise_hashes(query_fingerprint, **pair_searching_options)

    return query_hashes


def audio_identification(
        screen,
        path_to_queries,
        path_to_fingerprints,
        path_to_output,
        peak_picking_options={},
        pair_searching_options={}):

    curses.use_default_colors()
    print_status(screen, "blank_status", {})
    print_status(screen, "loading_db", {"db_file": path_to_fingerprints})

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
            screen,
            "analysing_file",
            { "now_analysing": entry.name })

        hash_start_time = time.perf_counter()
        # extract hashes from query
        query_hashes = get_query_hashes(
            entry.path,
            peak_picking_options,
            pair_searching_options)
        hash_time = time.perf_counter() - hash_start_time

        print_status(
            screen,
            "searching_db",
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
            screen,
            "finished_identifying",
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


def audioIdentification(
        path_to_queries,
        path_to_fingerprints,
        path_to_output,
        peak_picking_options={},
        pair_searching_options={}):
    
    curses.wrapper(
        audio_identification,
        path_to_queries,
        path_to_fingerprints,
        path_to_output,
        peak_picking_options,
        pair_searching_options) 
