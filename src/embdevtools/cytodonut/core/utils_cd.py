import multiprocessing as mp
import os
import pickle
import time
import warnings
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull
from tifffile import TiffFile

np.seterr(all="warn")


def get_file_name(path_data, f):
    """
    Parameters
    ----------
    path_data : str
        The path to the directory containing emb
    f : str or int
        if str returns path_data/emb
        if int returns the emb element in path_data

    Returns
    -------
    file, name
        full file path and file name.
    """
    files = os.listdir(path_data)
    if isinstance(f, str):
        for i, file in enumerate(files):
            if f in file:
                fid = i
    else:
        fid = f
    file = files[fid]
    name = file.split(".")[0]
    return file, name


def intersect2D(a, b):
    """
    Find row intersection between 2D numpy arrays, a and b.
    Returns another numpy array with shared rows
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


def get_only_unique(x):
    # Consider each row as indexing tuple & get linear indexing value
    lid = np.ravel_multi_index(x.T, x.max(0) + 1)

    # Get counts and unique indices
    _, idx, count = np.unique(lid, return_index=True, return_counts=True)

    # See which counts are exactly 1 and select the corresponding unique indices
    # and thus the correspnding rows from input as the final output
    out = x[idx[count == 1]]
    return out


def sefdiff2D(a, b):
    a_rows = a.view([("", a.dtype)] * a.shape[1])
    b_rows = b.view([("", b.dtype)] * b.shape[1])
    mask = np.setdiff1d(a_rows, b_rows).view(a.dtype).reshape(-1, a.shape[1])
    return mask


def sort_xy(x, y, ang_tolerance=0.2):
    x0 = np.mean(x)
    y0 = np.mean(y)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    warnings.filterwarnings("error")
    with warnings.catch_warnings():
        try:
            _angles = np.arccos((x - x0) / r)
            angles = [
                _angles[xid] + np.pi if _x > x0 else _angles[xid]
                for xid, _x in enumerate(x)
            ]
        except RuntimeWarning:
            return (x, y, False)

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    Difs = [
        np.abs(ang1 - ang2)
        for aid1, ang1 in enumerate(angles)
        for aid2, ang2 in enumerate(angles[aid1 + 1 :])
        if aid1 != aid2
    ]
    difs = np.nanmean(Difs)
    if difs > ang_tolerance:
        return (x_sorted, y_sorted, True)
    else:
        return (x_sorted, y_sorted, False)


def extract_ICM_TE_labels(cells, t, z):
    centers = []

    for cell in cells:
        if t not in cell.times:
            continue
        tid = cell.times.index(t)
        if z not in cell.zs[tid]:
            continue
        zid = cell.zs[tid].index(z)
        centers.append(cell.centers_all[tid][zid])

    centers = [cen[1:] for cen in centers if cen[0] == z]
    centers = np.array(centers)
    if len(centers) < 3:
        return [], []
    hull = ConvexHull(centers)
    outline = centers[hull.vertices]
    outline = np.array(outline).astype("int32")

    TE = []
    ICM = []
    for cell in cells:
        if t not in cell.times:
            continue
        tid = cell.times.index(t)
        if z not in cell.zs[tid]:
            continue
        zid = cell.zs[tid].index(z)
        if np.array(cell.centers_all[tid][zid][1:]).astype("int32") in outline:
            TE.append(cell.label)
        else:
            ICM.append(cell.label)
    return ICM, TE


def sort_points_counterclockwise(points):
    new_points = deepcopy(points)
    x = new_points[:, 1]
    y = new_points[:, 0]
    xsorted, ysorted, tolerance_bool = sort_xy(x, y)
    new_points[:, 1] = xsorted
    new_points[:, 0] = ysorted
    return new_points, tolerance_bool


def save_donuts(ES, path=None, filename=None):
    pthsave = path + filename
    file_to_store = open(pthsave + "_donuts.pickle", "wb")
    pickle.dump(ES, file_to_store)
    file_to_store.close()


def load_donuts(path=None, filename=None):
    pthsave = path + filename
    file_to_store = open(pthsave + "_donuts.pickle", "rb")
    donuts = pickle.load(file_to_store)
    file_to_store.close()
    return donuts


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


def printclear(n=1):
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def compute_ERK_traces(IMGS, cells, erkktr):
    for cell in cells:
        try:
            donuts = erkktr._get_donut(cell.label)
        except:
            continue
        ERKtrace = np.zeros_like(cell.times).astype("float64")
        for tid, t in enumerate(cell.times):
            erk = 0.0
            erkn = 0.0
            for zid, z in enumerate(cell.zs[tid]):
                img = IMGS[t, z]
                donut = donuts.donut_masks[tid][zid]
                xids = donut[:, 1]
                yids = donut[:, 0]
                erkdonutdist = img[xids, yids]

                nuclei = donuts.nuclei_masks[tid][zid]
                xids = nuclei[:, 1]
                yids = nuclei[:, 0]
                erknucleidist = img[xids, yids]
                erk += np.mean(erkdonutdist) / np.mean(erknucleidist)
                erkn += 1

            ERKtrace[tid] = erk / np.float(erkn)
        cell.ERKtrace = ERKtrace


def assign_fate(cells, times, slices):
    for cell in cells:
        cell.fate = [None for i in cell.times]
    for t in range(times):
        for z in range(slices):
            ICM, TE = extract_ICM_TE_labels(cells, t, z)

            for cell in cells:
                if t not in cell.times:
                    continue
                tid = cell.times.index(t)
                if z not in cell.zs[tid]:
                    continue
                if cell.label in ICM:
                    cell.fate[tid] = "ICM"
                elif cell.label in TE:
                    cell.fate[tid] = "TE"


def read_img_with_resolution(path_to_file, channel=0):
    with TiffFile(path_to_file) as tif:
        preIMGS = tif.asarray()
        shapeimg = preIMGS.shape
        if channel == None:
            if len(shapeimg) == 3:
                IMGS = np.array([tif.asarray()])
            else:
                IMGS = np.array(tif.asarray())
        else:
            if len(shapeimg) == 4:
                IMGS = np.array([tif.asarray()[:, channel, :, :]])
            else:
                IMGS = np.array(tif.asarray()[:, :, channel, :, :])
        imagej_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags
        # parse X, Y resolution
        npix, unit = tags["XResolution"].value
        xres = unit / npix
        npix, unit = tags["YResolution"].value
        yres = unit / npix
        assert xres == yres
        xyres = xres
        zres = imagej_metadata["spacing"]
    return IMGS, xyres, zres


def sort_coordinates(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(-angles)
    return list_of_xy_coords[indices]


def sort_outline(outline):
    hull = ConvexHull(outline)
    noutline = outline[hull.vertices]
    newoutline = sort_coordinates(noutline)
    return newoutline


import matplotlib.pyplot as plt


def plot_donuts(
    DONUTS,
    cells,
    IMGS_SEG,
    IMGS_ERK,
    t,
    z,
    labels="all",
    plot_outlines=True,
    plot_nuclei=True,
    plot_donut=True,
    EmbSeg=None,
):
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    imgseg = IMGS_SEG[t, z]
    imgerk = IMGS_ERK[t, z]

    ax[0].imshow(imgseg)
    ax[1].imshow(imgerk)

    if labels == "all":
        labels = [cell.label for cell in cells]

    for donut in DONUTS.Donuts:
        if donut.cell_label not in labels:
            continue
        cell = DONUTS._get_cell(cells, label=donut.cell_label)

        if t not in cell.times:
            continue
        tid = cell.times.index(t)
        if z not in cell.zs[tid]:
            continue
        zid = cell.zs[tid].index(z)

        outline = cell.outlines[tid][zid]
        mask = cell.masks[tid][zid]

        nuc_mask = donut.nuclei_masks[tid][zid]
        nuc_outline = sort_outline(donut.nuclei_outlines[tid][zid])
        don_mask = donut.donut_masks[tid][zid]
        maskout = donut.donut_outer_mask[tid][zid]
        don_outline_in = sort_outline(donut.donut_outlines_in[tid][zid])
        don_outline_out = sort_outline(donut.donut_outlines_out[tid][zid])

        if plot_outlines:
            ax[0].scatter(outline[:, 0], outline[:, 1], s=1, c="k", alpha=0.5)
            ax[0].plot(
                don_outline_in[:, 0],
                don_outline_in[:, 1],
                linewidth=1,
                c="orange",
                alpha=0.5,
            )  # , marker='o',markersize=1)
            ax[0].plot(
                don_outline_out[:, 0],
                don_outline_out[:, 1],
                linewidth=1,
                c="orange",
                alpha=0.5,
            )  # , marker='o',markersize=1)
            ax[0].plot(
                nuc_outline[:, 0], nuc_outline[:, 1], linewidth=1, c="purple", alpha=0.5
            )  # , marker='o',markersize=1)
            ax[1].scatter(outline[:, 0], outline[:, 1], s=1, c="k", alpha=0.5)
            ax[1].plot(
                don_outline_in[:, 0],
                don_outline_in[:, 1],
                linewidth=1,
                c="orange",
                alpha=0.5,
            )  # , marker='o',markersize=1)
            ax[1].plot(
                don_outline_out[:, 0],
                don_outline_out[:, 1],
                linewidth=1,
                c="orange",
                alpha=0.5,
            )  # , marker='o',markersize=1)
            ax[1].plot(
                nuc_outline[:, 0], nuc_outline[:, 1], linewidth=1, c="purple", alpha=0.5
            )  # , marker='o',markersize=1)

        if plot_nuclei:
            ax[1].scatter(nuc_mask[:, 0], nuc_mask[:, 1], s=1, c="green", alpha=1)
            ax[0].scatter(nuc_mask[:, 0], nuc_mask[:, 1], s=1, c="green", alpha=1)

        if plot_donut:
            ax[1].scatter(don_mask[:, 0], don_mask[:, 1], s=1, c="red", alpha=0.1)
            ax[0].scatter(don_mask[:, 0], don_mask[:, 1], s=1, c="red", alpha=0.1)

        xs = cell.centersi[tid][zid]
        ys = cell.centersj[tid][zid]
        label = cell.label
        ax[0].annotate(str(label), xy=(ys, xs), c="w")
        ax[0].scatter([ys], [xs], s=0.5, c="white")
        ax[0].axis(False)
        ax[1].annotate(str(label), xy=(ys, xs), c="w")
        ax[1].scatter([ys], [xs], s=0.5, c="white")
        ax[1].axis(False)

    if EmbSeg is not None:
        ax[1].scatter(
            EmbSeg.Embmask[t][z][:, 0],
            EmbSeg.Embmask[t][z][:, 1],
            s=1,
            c="blue",
            alpha=0.05,
        )
        ax[0].scatter(
            EmbSeg.Embmask[t][z][:, 0],
            EmbSeg.Embmask[t][z][:, 1],
            s=1,
            c="blue",
            alpha=0.05,
        )

    plt.tight_layout()
    plt.show()


class PlotAction:
    def __init__(self, fig, ax, donuts, IMGS1, IMGS2, id, mode):
        self.fig = fig
        self.ax = ax
        self.id = id
        self.donuts = donuts
        self.list_of_cells = []
        self.act = fig.canvas.mpl_connect("key_press_event", self)
        self.ctrl_press = self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key_press
        )
        self.ctrl_release = self.fig.canvas.mpl_connect(
            "key_release_event", self.on_key_release
        )
        self.ctrl_is_held = False
        self.current_state = None
        self.current_subplot = None
        self.cr = 0
        self.t = 0
        self.zs = []
        self.z = None
        self.scl = fig.canvas.mpl_connect("scroll_event", self.onscroll)
        groupsize = 1
        self.times = IMGS1.shape[0]
        self.slices = IMGS1.shape[1]
        self.max_round = self.slices
        self.get_size()
        self.mode = mode
        self.plot_outlines = True

    def __call__(self, event):
        # To be defined
        pass

    def on_key_press(self, event):
        if event.key == "control":
            self.ctrl_is_held = True

    def on_key_release(self, event):
        if event.key == "control":
            self.ctrl_is_held = False

    # The function to be called anytime a t-slider's value changes
    def update_slider_t(self, t):
        self.t = t - 1
        # REPLOT()
        self.update()

    # The function to be called anytime a z-slider's value changes
    def update_slider_z(self, cr):
        self.cr = cr
        # REPLOT()
        self.update()

    def onscroll(self, event):
        if self.ctrl_is_held:
            # if self.current_state == None: self.current_state="SCL"
            if event.button == "up":
                self.t = self.t + 1
            elif event.button == "down":
                self.t = self.t - 1

            self.t = max(self.t, 0)
            self.t = min(self.t, self.CT.times - 1)
            self.CT._time_sliders[self.id].set_val(self.t + 1)
            # REPLOT()
            self.update()

            if self.current_state == "SCL":
                self.current_state = None

        else:
            if event.button == "up":
                self.cr = self.cr - 1
            elif event.button == "down":
                self.cr = self.cr + 1

            self.cr = max(self.cr, 0)
            self.cr = min(self.cr, self.max_round)
            self.CT._z_sliders[self.id].set_val(self.cr)

            self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
            self.update()

            if self.current_state == "SCL":
                self.current_state = None

    def get_size(self):
        bboxfig = self.fig.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        widthfig, heightfig = (
            bboxfig.width * self.fig.dpi,
            bboxfig.height * self.fig.dpi,
        )
        self.figwidth = widthfig
        self.figheight = heightfig

    def reploting(self):
        self.CT.replot_tracking(self, plot_outlines=self.plot_outlines)
        self.fig.canvas.draw_idle()
        self.fig.canvas.draw()

    def update(self):
        pass
