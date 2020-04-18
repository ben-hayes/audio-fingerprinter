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


if __name__ == "__main__":
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

        db_name = "param_search/fingerprint_db_%d.db" % n
        output_name = "param_search/identified_tracks_%d.txt" % n

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

        with open("param_search/params_%d.json" % n, "w") as f:
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
    
    with open("param_search/output.txt", "w") as f:
        for n, (sc, t) in enumerate(zip(performance, times)):
            line = "Attempt #%d — Score: %.3f, Time: %.3f" % (n, sc, t)
            f.write(line)
            print(line)

    print ("Finished: %dth attempt best" % (argmax(performance)))
    print ("Finished: %dth attempt fastest" % (argmin(times)))
    print ("Finished: %dth attempt best ratio" % (argmax(ratio)))