import json

from numpy import argmax
from numpy.random import randint

from fingerprint_builder import fingerprintBuilder
from audio_identification import audioIdentification

N_ATTEMPTS = 10

available_parameters = {
    "peak_picking": {
        "kappa": {
            "min": 53,
            "max": 73
        },
        "tau": {
            "min": 6,
            "max": 26
        },
        "hop_kappa": {
            "min": 58,
            "max": 78
        },
        "hop_tau": {
            "min": 6,
            "max": 26
        }
    },
    "pair_searching": {
        "target_time_offset": {
            "min": 2,
            "max": 22
        },
        "target_time_width": {
            "min": 188,
            "max": 198
        },
        "target_freq_height": {
            "min": 200,
            "max": 220
        }
    }
}

if __name__ == "__main__":
    performance = []
    for n in range(N_ATTEMPTS):
        print("########### RANDOM SEARCH ATTEMPT %d ###########" % (n + 1))
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
        fingerprintBuilder(
            "data/clean_subset",
            db_name,
            peak_picking_options,
            pair_searching_options)
        score = audioIdentification(
            "data/query_subset",
            db_name,
            output_name,
            peak_picking_options=peak_picking_options,
            pair_searching_options=pair_searching_options)

        performance.append(score)

        with open("param_search/params_%d.json" % n, "w") as f:
            json.dump({
                    "peak_picking_options": peak_picking_options,
                    "pair_searching_options": pair_searching_options
                },
                f
            )
        print("########### FINISHED RANDOM SEARCH ATTEMPT %d ###########" % (n + 1))
        print("########### SCORE %.3f ###########" % score)
    for n, sc in enumerate(performance):
        print("Attempt #%d — Score: %.3f" % (n + 1, sc))

    print ("Finished: %dth attempt best" % (argmax(performance) + 1))