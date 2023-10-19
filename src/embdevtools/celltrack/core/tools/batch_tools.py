import math

def compute_batch_times(round, batch_size, batch_overlap, totalsize):
    first = (batch_size * round) - (batch_overlap * round)
    last = first + batch_size
    last = min(last, totalsize)
    return first, last

def extract_files(path):
    total_times = 0
    return total_times