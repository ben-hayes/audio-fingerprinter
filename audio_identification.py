import json
import os
import pickle

import numpy as np

from fingerprint_builder import create_fingerprint

def audioIdentification(path_to_queries, path_to_fingerprints, path_to_output):
    output_file = open(path_to_output, "w")
    fingerprints = {}
    for fingerprint_entry in os.scandir(path_to_fingerprints):
        with open(fingerprint_entry.path, "rb") as f:
            document_inverted_lists = pickle.load(f)
        document_name, _ = os.path.splitext(fingerprint_entry.name)
        fingerprints[document_name] = document_inverted_lists

    scores = {}
    for entry in os.scandir(path_to_queries):
        print("Analysing %s...                 " % entry.name, end="\r")
        if os.path.splitext(entry.name)[1] != ".wav":
            continue

        query_fingerprint = create_fingerprint(entry.path)
        query_points = np.dstack(np.nonzero(query_fingerprint)).squeeze()

        scores[entry.name] = {}
        for document_name in fingerprints:
            print("Comparing %s to %s...                " % (entry.name, document_name), end="\r")
            document_inverted_lists = fingerprints[document_name]
            shifted_lists = [document_inverted_lists[h] - n\
                             for h, n in query_points]

            min_m = np.min([s.min() for s in shifted_lists if len(s) > 0])
            max_m = np.max([s.max() for s in shifted_lists if len(s) > 0])
            m_values = np.arange(min_m, max_m + 1)
            m_scores = np.array([np.in1d(m_values, shifted_list) for shifted_list in shifted_lists]).astype(int)
            m_scores = m_scores.sum(axis=0)

            scores[entry.name][document_name] = int(np.max(m_scores))

        sorted_scores = [key for key in scores[entry.name]]
        sorted_scores = sorted(sorted_scores, key=lambda x: scores[entry.name][x])

        print("Finished comparing %s. Best scores were %s.                " % (entry.name, ", ".join(sorted_scores[:3])), end="\n")

        output_line = "%s %s.wav %s.wav %s.wav\n" % (entry.name, *sorted_scores[:3])
        output_file.write(output_line)
    output_file.close()


