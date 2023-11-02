from .input_tools import get_file_names
import numpy as np

def compute_batch_times(round, batch_size, batch_overlap, totalsize):
    first = (batch_size * round) - (batch_overlap * round)
    last = first + batch_size
    last = min(last, totalsize)
    return first, last

def extract_total_times_from_files(path):
    total_times = 0
    files = get_file_names(path)
    for file in files:
        try:
            _ = int(file.split(".")[0])
            total_times += 1
        except:
            continue

    return total_times


def check_and_fill_batch_args(batch_args):
    new_batch_args = {
        "batch_size": 5,
        "batch_overlap": 1,
    }
    if batch_args["batch_size"] <= batch_args["batch_overlap"]:
        raise Exception("batch size has to be bigger than batch overlap")
    for sarg in batch_args.keys():
        try:
            new_batch_args[sarg] = batch_args[sarg]
        except KeyError:
            raise Exception(
                "key %s is not a correct batch argument"
                % sarg
            )

    return new_batch_args
