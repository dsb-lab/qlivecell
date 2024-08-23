import multiprocessing as mp
import time


def worker(input, output):
    # The input are the arguments of the function

    # The output is the ERKKTR_donut class

    for func, args in iter(input.get, "STOP"):
        result = func(*args)
        output.put(result)


def multiprocess(threads, worker, TASKS, daemon=None):
    task_queue, done_queue = multiprocess_start(threads, worker, TASKS, daemon=None)
    results = multiprocess_get_results(done_queue, TASKS)
    multiprocess_end(task_queue)
    return results


def multiprocess_start(threads, worker, TASKS, daemon=None):
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    # Submit tasks
    for task in TASKS:
        task_queue.put(task)

    # Start worker processes
    for i in range(threads):
        p = mp.Process(target=worker, args=(task_queue, done_queue))
        if daemon is not None:
            p.daemon = daemon
        p.start()

    return task_queue, done_queue


def multiprocess_end(task_queue):
    # Tell child processes to stop

    iii = 0
    while len(mp.active_children()) > 0:
        if iii != 0:
            time.sleep(0.1)
        for process in mp.active_children():
            # Send STOP signal to our task queue
            task_queue.put("STOP")

            # Terminate process
            process.terminate()
            process.join()
        iii += 1


def multiprocess_add_tasks(task_queue, TASKS):
    # Submit tasks
    for task in TASKS:
        task_queue.put(task)

    return task_queue


def multiprocess_get_results(done_queue, TASKS):
    results = [done_queue.get() for t in TASKS]

    return results
