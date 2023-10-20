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