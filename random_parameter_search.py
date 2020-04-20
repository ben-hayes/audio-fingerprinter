"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 2: Audio Identification

File: random_parameter_search.py
Description: A simple script for performing a random parameter search across
             the parameter space of the fingerprinting and identification
             algorithms.
             
"""
from argparse import ArgumentParser
import json
import time

import numpy as np
from numpy import argmax, argmin
from numpy.random import randint

from fingerprint_builder import fingerprintBuilder
from audio_identification import audioIdentification
from evaluation import parse_id_file, mean_avg_precision

N_ATTEMPTS = 100


available_parameters = {
    "peak_picking": {
        "kappa": {
            "min": 8,
            "max": 100
        },
        "tau": {
            "min": 8,
            "max": 100
        },
        "hop_kappa": {
            "min": 4,
            "max": 100
        },
        "hop_tau": {
            "min": 4,
            "max": 100
        }
    },
    "pair_searching": {
        "target_time_offset": {
            "min": 2,
            "max": 100
        },
        "target_time_width": {
            "min": 8,
            "max": 100
        },
        "target_freq_height": {
            "min": 8,
            "max": 100
        }
    }
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_folder", default="param_search")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    performance = []
    times = []
    for n in range(N_ATTEMPTS):
        print("########### RANDOM SEARCH ATTEMPT %d ###########" % (n))
        peak_picking_options = {
            key: randint(
                available_parameters["peak_picking"][key]["min"],
                available_parameters["peak_picking"][key]["max"] + 1)
            for key in available_parameters["peak_picking"] 
        }
        peak_picking_options["hop_kappa"] =\
            randint(
                available_parameters["peak_picking"]["hop_kappa"]["min"],
                min(
                    available_parameters["peak_picking"]["hop_kappa"]["max"],
                    peak_picking_options["kappa"] * 2
                )
            )
        peak_picking_options["hop_tau"] =\
            randint(
                available_parameters["peak_picking"]["hop_tau"]["min"],
                min(
                    available_parameters["peak_picking"]["hop_tau"]["max"],
                    peak_picking_options["tau"] * 2
                )
            )

        pair_searching_options = {
            key: randint(
                available_parameters["pair_searching"][key]["min"],
                available_parameters["pair_searching"][key]["max"] + 1)
            for key in available_parameters["pair_searching"] 
        }
        print(json.dumps(peak_picking_options, indent=4))
        print(json.dumps(pair_searching_options, indent=4))

        db_name = "%s/fingerprint_db_%d.db" % (args.output_folder, n)
        output_name = "%s/identified_tracks_%d.txt" % (args.output_folder, n)

        start_time = time.perf_counter()
        fingerprintBuilder(
            "data/clean_subset",
            db_name,
            peak_picking_options,
            pair_searching_options)
        audioIdentification(
            "data/query_subset",
            db_name,
            output_name,
            peak_picking_options=peak_picking_options,
            pair_searching_options=pair_searching_options)
        end_time = time.perf_counter()
        
        relevances = parse_id_file(output_name)
        score = mean_avg_precision(relevances)

        performance.append(score)
        times.append(end_time - start_time)

        with open("%s/params_%d.json" % (args.output_folder, n), "w") as f:
            json.dump({
                    "peak_picking_options": peak_picking_options,
                    "pair_searching_options": pair_searching_options
                },
                f
            )
        print("########### FINISHED RANDOM SEARCH ATTEMPT %d ###########" % (n))
        print("########### SCORE %.3f ###########" % score)
        print("########### TIME  %.3f ###########" % (end_time - start_time))
    
    ratio = np.array(performance) / np.array(times)
    
    with open("%s/output.txt" % args.output_folder, "w") as f:
        for n, (sc, t) in enumerate(zip(performance, times)):
            line = "Attempt #%d — Score: %.3f, Time: %.3f" % (n, sc, t)
            f.write(line)
            print(line)

    print ("Finished: %dth attempt best" % (argmax(performance)))
    print ("Finished: %dth attempt fastest" % (argmin(times)))
    print ("Finished: %dth attempt best ratio" % (argmax(ratio)))