from argparse import ArgumentParser

import numpy as np

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("input_file")

    return parser.parse_args()


def numpy_array_from_uneven_lists(lists):
    longest_list = len(max(lists, key=len))
    np_array = np.zeros((len(lists), longest_list))
    for i, x in enumerate(lists):
        np_array[i, :len(x)] = x
    return np_array


def parse_id_file(id_file):
    with open(id_file) as f:
        lines = [line.split() for line in f]

    relevances = []
    for line in lines:
        split_line = line[0].split("-")
        ground_truth = split_line[0]
        relevances.append(
            [1 if ground_truth in doc else 0 for doc in line[1:]])

    return numpy_array_from_uneven_lists(relevances)


def precision(rank, relevances):
    return np.sum(relevances[:,:rank], axis=1) / float(rank)


def recall(rank, relevances, num_relevant_docs=1):
    return np.sum(relevances[:,:rank], axis=1) / float(num_relevant_docs)


def f_measure(rank, relevances, num_relevant_docs=1):
    prec = precision(rank, relevances)
    rec = recall(rank, relevances, num_relevant_docs)
    f = (2 * prec * rec) / (prec + rec)
    return np.nan_to_num(f)


def avg_precision(relevances, num_relevant_docs=1):
    if relevances.shape[-1] == 0:
        return 0.0
    inner_sum = np.concatenate(
            [precision(r + 1, relevances).reshape(-1, 1) for r in range(
                relevances.shape[-1])],
            axis=1)\
        * relevances[:, :relevances.shape[-1]]
    return np.sum(inner_sum, axis=1) / float(num_relevant_docs)


def mean_avg_precision(relevances, num_relevant_docs=1):
    return np.mean(avg_precision(relevances, num_relevant_docs))


if __name__ == "__main__":
    args = parse_args()

    relevance_function = parse_id_file(args.input_file)
    for r in range(1, 3 + 1):
        print("---- Rank %d ----" % r)
        print(
            "Mean Precision: %.3f" % np.mean(precision(r, relevance_function)))
        print("Mean Recall: %.3f" % np.mean(recall(r, relevance_function)))
        print(
            "Mean f-measure: %.3f" % np.mean(f_measure(r, relevance_function)))
    print("----------------")
    print(
        "Mean avg precision: %.3f" % mean_avg_precision(relevance_function))