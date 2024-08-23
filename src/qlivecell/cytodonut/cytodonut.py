from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from .core.tools.cell_tools import (compute_distance_cell,
                                    extract_all_XYZ_positions_cell,
                                    update_cell)
from .core.utils_cd import *


def comptute_donut_masks(donut, cell_masks):
    donut.compute_donut_masks(cell_masks)


class ERKKTR_donut:
    def __init__(
        self,
        cell,
        innerpad=1,
        outterpad=1,
        donut_width=1,
        min_outline_length=50,
        inhull_method="delaunay",
    ):
        self.inpad = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self._min_outline_length = min_outline_length
        if inhull_method == "delaunay":
            self._inhull = self._inhull_Delaunay
        elif inhull_method == "cross":
            self._inhull = self._inhull_cross
        elif inhull_method == "linprog":
            self._inhull = self._inhull_linprog
        else:
            self._inhull = self._inhull_Delaunay
        self.cell_label = cell.label
        self._masks_computed = False
        self.compute_donut_outlines(cell)
        self.compute_donut_masks(cell.masks)
        self._masks_computed = True
        self.compute_nuclei_mask(cell)

    def compute_nuclei_mask(self, cell):
        self.nuclei_masks = deepcopy(cell.masks)
        self.nuclei_outlines = deepcopy(cell.outlines)
        for tid, t in enumerate(cell.times):
            for zid, z in enumerate(cell.zs[tid]):
                outline = cell.outlines[tid][zid]
                newoutline, midx, midy = self._expand_hull(outline, inc=-self.inpad)
                newoutline = self._increase_point_resolution(newoutline)
                _hull = ConvexHull(newoutline)
                newoutline = newoutline[_hull.vertices]
                hull = Delaunay(newoutline)

                self.nuclei_outlines[tid][zid] = np.array(newoutline).astype("int32")
                self.nuclei_masks[tid][zid] = self._points_within_hull(
                    hull, self.nuclei_outlines[tid][zid]
                )

    def compute_donut_outlines(self, cell):
        self.donut_outlines_in = deepcopy(cell.outlines)
        self.donut_outlines_out = deepcopy(cell.outlines)
        for tid, t in enumerate(cell.times):
            for zid, z in enumerate(cell.zs[tid]):
                outline = cell.outlines[tid][zid]
                hull = ConvexHull(outline)
                outline = outline[hull.vertices]
                outline = np.array(outline).astype("int32")

                inneroutline, midx, midy = self._expand_hull(outline, inc=self.outpad)
                outteroutline, midx, midy = self._expand_hull(
                    outline, inc=self.outpad + self.dwidht
                )

                # inneroutline=self._increase_point_resolution(inneroutline)
                # outteroutline=self._increase_point_resolution(outteroutline)

                _hull_in = ConvexHull(inneroutline)
                inneroutline = inneroutline[_hull_in.vertices]
                inneroutline = np.array(inneroutline).astype("int32")

                _hull_out = ConvexHull(outteroutline)
                outteroutline = outteroutline[_hull_out.vertices]
                outteroutline = np.array(outteroutline).astype("int32")

                self.donut_outlines_in[tid][zid] = inneroutline
                self.donut_outlines_out[tid][zid] = outteroutline

    def compute_donut_masks(self, cell_masks):
        if not self._masks_computed:
            self.donut_masks = deepcopy(cell_masks)
            self.donut_outer_mask = deepcopy(cell_masks)
            self.donut_inner_mask = deepcopy(cell_masks)
        for tid in range(len(self.donut_masks)):
            for zid in range(len(self.donut_masks[tid])):
                self.compute_donut_mask(tid, zid)

    def compute_donut_mask(self, tid, zid):
        inneroutline = self.donut_outlines_in[tid][zid]
        outteroutline = self.donut_outlines_out[tid][zid]

        # THIS NEEDS TO BE REVISED
        if inneroutline is None:
            return
        if len(inneroutline) < 4:
            return
        hull_in = Delaunay(inneroutline)
        if len(outteroutline) < 4:
            return
        hull_out = Delaunay(outteroutline)

        maskin = self._points_within_hull(hull_in, inneroutline)
        maskout = self._points_within_hull(hull_out, outteroutline)

        mask = sefdiff2D(maskout, maskin)
        self.donut_outer_mask[tid][zid] = np.array(maskout)
        self.donut_inner_mask[tid][zid] = np.array(maskin)
        self.donut_masks[tid][zid] = np.array(mask)
        return

    def sort_points_counterclockwise(self, points):
        x = points[:, 1]
        y = points[:, 0]

        xsorted, ysorted, tolerance_bool = sort_xy(x, y)
        points[:, 1] = xsorted
        points[:, 0] = ysorted
        return points, tolerance_bool

    def _expand_hull(self, outline, inc=1):
        newoutline = []
        midpointx = (max(outline[:, 0]) + min(outline[:, 0])) / 2
        midpointy = (max(outline[:, 1]) + min(outline[:, 1])) / 2

        for p in outline:
            newp = [0, 0]

            # Get angle between point and center
            x = p[0] - midpointx
            y = p[1] - midpointy
            theta = np.arctan2(y, x)
            xinc = inc * np.cos(theta)
            yinc = inc * np.sin(theta)
            newp[0] = x + xinc + midpointx
            newp[1] = y + yinc + midpointy
            newoutline.append(newp)
        return np.array(newoutline), midpointx, midpointy

    def _inhull_Delaunay(self, hull, p):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """

        return hull.find_simplex(p) >= 0

    def _points_within_hull(self, hull, outline):
        # With this function we compute the points contained within a hull or outline.
        pointsinside = []
        maxx = max(outline[:, 1])
        maxy = max(outline[:, 0])
        minx = min(outline[:, 1])
        miny = min(outline[:, 0])
        xrange = range(minx, maxx)
        yrange = range(miny, maxy)
        for i in yrange:
            for j in xrange:
                p = [i, j]
                if self._inhull(hull, p):
                    pointsinside.append(p)

        return np.array(pointsinside)

    def _increase_point_resolution(self, outline):
        if len(outline) < 3:
            return outline
        rounds = np.ceil(np.log2(self._min_outline_length / len(outline))).astype(
            "int32"
        )
        if rounds <= 0:
            newoutline_new = np.copy(outline)
        for r in range(rounds):
            if r == 0:
                pre_outline = np.copy(outline)
            else:
                pre_outline = np.copy(newoutline_new)
            newoutline_new = np.copy(pre_outline)
            i = 0
            while i < len(pre_outline) * 2 - 2:
                newpoint = np.array(
                    [
                        np.rint((newoutline_new[i] + newoutline_new[i + 1]) / 2).astype(
                            "int32"
                        )
                    ]
                )
                newoutline_new = np.insert(newoutline_new, i + 1, newpoint, axis=0)
                i += 2
            newpoint = np.array(
                [np.rint((pre_outline[-1] + pre_outline[0]) / 2).astype("int32")]
            )
            newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

        return newoutline_new

    def correct_donut_embryo_overlap_c(self, ti, zi, mask_emb, label):
        oi_out = self.donut_outlines_out[ti][zi]
        oi_inn = self.donut_outlines_in[ti][zi]
        maskout_cell = self.donut_outer_mask[ti][zi]
        maskout_cell = np.vstack((maskout_cell, oi_out))
        maskout_intersection = intersect2D(maskout_cell, mask_emb)

        # Check intersection with OUTTER outline

        oi_mc_intersection = intersect2D(oi_out, maskout_intersection)
        if len(oi_mc_intersection) < 4:
            return (label, None, None)
        new_oi_out, tolerance_bool1 = sort_points_counterclockwise(oi_mc_intersection)
        # Check intersection with INNER outline

        oi_mc_intersection = intersect2D(oi_inn, maskout_intersection)
        if len(oi_mc_intersection) < 4:
            return (label, None, None)
        new_oi_in, tolerance_bool2 = sort_points_counterclockwise(oi_mc_intersection)

        if not tolerance_bool1 or not tolerance_bool2:
            return (label, None, None)
        return (label, new_oi_out, new_oi_in)


def recompute_donut_masks(label, cell_masks, out_outlines, in_outlines):
    donut_masks = deepcopy(cell_masks)
    donut_outer_mask = deepcopy(cell_masks)
    donut_inner_mask = deepcopy(cell_masks)
    for tid in range(len(donut_masks)):
        for zid in range(len(donut_masks[tid])):
            d_m, d_o_m, d_i_m = recompute_donut_mask(
                tid, zid, out_outlines, in_outlines, label
            )
            donut_masks[tid][zid] = d_m
            donut_outer_mask[tid][zid] = d_o_m
            donut_inner_mask[tid][zid] = d_i_m
    return (label, donut_masks, donut_outer_mask, donut_inner_mask)


def recompute_donut_mask(tid, zid, donut_outlines_out, donut_outlines_in, label):
    inneroutline = donut_outlines_in[tid][zid]
    outteroutline = donut_outlines_out[tid][zid]

    if inneroutline is None or outteroutline is None:
        return (None, None, None)
    hull_in = Delaunay(inneroutline)
    hull_out = Delaunay(outteroutline)

    maskin = points_within_hull(hull_in, inneroutline)
    maskout = points_within_hull(hull_out, outteroutline)

    mask = sefdiff2D(maskout, maskin)
    return (np.array(mask), np.array(maskout), np.array(maskin))


def inhull_Delaunay(hull, p):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    return hull.find_simplex(p) >= 0


def points_within_hull(hull, outline):
    # With this function we compute the points contained within a hull or outline.
    pointsinside = []
    maxx = max(outline[:, 1])
    maxy = max(outline[:, 0])
    minx = min(outline[:, 1])
    miny = min(outline[:, 0])
    xrange = range(minx, maxx)
    yrange = range(miny, maxy)
    for i in yrange:
        for j in xrange:
            p = [i, j]
            if inhull_Delaunay(hull, p):
                pointsinside.append(p)

    return np.array(pointsinside)


class ERKKTR:
    def __init__(
        self,
        IMGS,
        innerpad=1,
        outterpad=2,
        donut_width=4,
        min_outline_length=50,
        cell_distance_th=70.0,
        mp_threads=None,
    ):
        self.inpad = innerpad
        self.outpad = outterpad
        self.dwidht = donut_width
        self.min_outline_length = min_outline_length
        self.times = IMGS.shape[0]
        self.slices = IMGS.shape[1]
        self._dist_th = cell_distance_th
        self.Donuts = []
        if mp_threads == "all":
            self._threads = mp.cpu_count() - 1
        else:
            self._threads = mp_threads
        self.list_of_cells = range(10**4)

    def execute_erkktr(
        self, cell, innerpad, outterpad, donut_width, min_outline_length
    ):
        return ERKKTR_donut(
            cell, innerpad, outterpad, donut_width, min_outline_length, "delaunay"
        )

    def create_donuts(
        self,
        cells,
        IMGS,
        EmbSeg,
        innerpad=None,
        outterpad=None,
        donut_width=None,
        change_threads=False,
    ):
        if innerpad is None:
            innerpad = self.inpad
        if outterpad is None:
            outterpad = self.outpad
        if donut_width is None:
            donut_width = self.dwidht
        if change_threads is False:
            threads = self._threads
        else:
            if change_threads == "all":
                threads = mp.cpu_count() - 1
            else:
                threads = change_threads

        # Check for multi or single processing
        for cell in cells:
            update_cell(cell, IMGS)
        if threads is None:
            tcells = len(cells)
            for celli, cell in enumerate(cells):
                print("cell %d / %d" % (celli + 1, tcells))
                if cell.label not in self.list_of_cells:
                    continue
                self.Donuts.append(
                    self.execute_erkktr(
                        cell, innerpad, outterpad, donut_width, self.min_outline_length
                    )
                )
        else:
            TASKS = [
                (
                    self.execute_erkktr,
                    (cell, innerpad, outterpad, donut_width, self.min_outline_length),
                )
                for cell in cells
            ]
            results = multiprocess(self._threads, worker, TASKS)
            # Post processing of outputs
            for ed in results:
                lab = ed.cell_label
                self.Donuts.append(ed)

        print("Running donut corrections...")
        print("...correcting cell-cell overlap")
        self.correct_cell_to_cell_overlap(cells)
        print("...correcting donut-embryo overlap")
        self.correct_donut_embryo_overlap(cells, EmbSeg)
        self.remove_empty_outlines(cells)
        print("...correcting nuclei-donut overlap and computing masks")
        self.correct_donut_nuclei_overlap(cells)
        print("...corrections finished")
        return

    def remove_empty_outlines(self, cells):
        donuts_id_rem = []
        for d, donut in enumerate(self.Donuts):
            cell = self._get_cell(cells, donut.cell_label)
            for tid, t in enumerate(cell.times):
                for zid, z in enumerate(cell.zs[tid]):
                    if donut.donut_outlines_out[tid][zid] is None:
                        if d not in donuts_id_rem:
                            donuts_id_rem.append(d)

        donuts_id_rem.sort(reverse=True)
        for d in donuts_id_rem:
            self.Donuts.pop(d)

    def correct_cell_to_cell_overlap(self, cells):
        print()
        for _, t in enumerate(range(self.times)):
            for _, z in enumerate(range(self.slices)):
                printclear(n=1)
                print("t = %d/%d , z = %d/%d" % (t + 1, self.times, z + 1, self.slices))

                Cells = []
                for cell in cells:
                    if cell.label not in self.list_of_cells:
                        continue
                    if t not in cell.times:
                        continue
                    ti = cell.times.index(t)
                    if z not in cell.zs[ti]:
                        continue
                    Cells.append(cell)
                self.correct_cell_to_cell_overlap_z(Cells, t, z, self._dist_th)
        printclear()
        return

    def correct_cell_to_cell_overlap_z(self, Cells, t, z, dist_th):
        for cell_i in Cells:
            donut_i = self._get_donut(cell_i.label)
            cells_close = []
            ti = cell_i.times.index(t)
            zi = cell_i.zs[ti].index(z)

            for cell_j_id, cell_j in enumerate(Cells):
                if cell_i.label == cell_j.label:
                    continue
                dist = compute_distance_cell(cell_i, cell_j, t, z, axis="xy")
                if dist < dist_th:
                    cells_close.append(cell_j_id)

            if len(cells_close) == 0:
                continue
            # Now for the the closest ones we check for overlaping
            oi_out = donut_i.donut_outlines_out[ti][zi]
            oi_out = donut_i._increase_point_resolution(oi_out)

            oi_inn = donut_i.donut_outlines_in[ti][zi]
            oi_inn = donut_i._increase_point_resolution(oi_inn)

            maskout_cell_i = donut_i.donut_outer_mask[ti][zi]
            maskout_cell_i = np.vstack((maskout_cell_i, oi_out))

            # For each of the close cells, compute intersection of outer donut masks

            for cell_j_id in cells_close:
                cell_j = Cells[cell_j_id]
                donut_j = self._get_donut(Cells[cell_j_id].label)

                tcc = cell_j.times.index(t)
                zcc = cell_j.zs[tcc].index(z)
                maskout_cell_j = donut_j.donut_outer_mask[tcc][zcc]

                oj_out = donut_j.donut_outlines_out[tcc][zcc]
                oj_out = donut_j._increase_point_resolution(oj_out)

                maskout_cell_j = np.vstack((maskout_cell_j, oj_out))

                maskout_intersection = intersect2D(maskout_cell_i, maskout_cell_j)

                if len(maskout_intersection) == 0:
                    continue

                # Check intersection with OUTTER outline

                # Get intersection between outline and the masks intersection
                # These are the points to be removed from the ouline
                oi_mc_intersection = intersect2D(oi_out, maskout_intersection)
                if len(oi_mc_intersection) != 0:
                    new_oi = get_only_unique(np.vstack((oi_out, oi_mc_intersection)))
                    if len(new_oi) != 0:
                        new_oi, tolerance_bool = donut_i.sort_points_counterclockwise(
                            new_oi
                        )
                        donut_i.donut_outlines_out[ti][zi] = deepcopy(new_oi)
                    else:
                        pass
                        # print()
                        # print(donut_j.cell_label)
                        # print(donut_i.cell_label)
                        # print(tcc)
                        # print(t)
                        # print(zcc)
                        # print(z)

                oj_mc_intersection = intersect2D(oj_out, maskout_intersection)
                if len(oj_mc_intersection) != 0:
                    new_oj = get_only_unique(np.vstack((oj_out, oj_mc_intersection)))
                    if len(new_oj) != 0:
                        new_oj, tolerance_bool = donut_j.sort_points_counterclockwise(
                            new_oj
                        )
                        donut_j.donut_outlines_out[tcc][zcc] = deepcopy(new_oj)
                    else:
                        pass
                        # print()
                        # print(donut_j.cell_label)
                        # print(donut_i.cell_label)
                        # print(tcc)
                        # print(t)
                        # print(zcc)
                        # print(z)
                # Check intersection with INNER outline
                oj_inn = donut_j.donut_outlines_in[tcc][zcc]
                oj_inn = donut_j._increase_point_resolution(oj_inn)

                # Get intersection between outline and the masks intersection
                # These are the points to be removed from the ouline
                oi_mc_intersection = intersect2D(oi_inn, maskout_intersection)
                if len(oi_mc_intersection) != 0:
                    new_oi = get_only_unique(np.vstack((oi_inn, oi_mc_intersection)))

                    if len(new_oi) != 0:
                        new_oi, tolerance_bool = donut_i.sort_points_counterclockwise(
                            new_oi
                        )
                        donut_i.donut_outlines_in[ti][zi] = deepcopy(new_oi)
                    else:
                        pass
                        # print()
                        # print(donut_j.cell_label)
                        # print(donut_i.cell_label)
                        # print(tcc)
                        # print(t)
                        # print(zcc)
                        # print(z)

                oj_mc_intersection = intersect2D(oj_inn, maskout_intersection)
                if len(oj_mc_intersection) != 0:
                    new_oj = get_only_unique(np.vstack((oj_inn, oj_mc_intersection)))
                    if len(new_oj) != 0:
                        new_oj, tolerance_bool = donut_j.sort_points_counterclockwise(
                            new_oj
                        )
                        donut_j.donut_outlines_in[tcc][zcc] = deepcopy(new_oj)
                    else:
                        pass
        return None

    def correct_donut_embryo_overlap(self, cells, EmbSeg):
        # if self._threads is not None: task_queue, done_queue = multiprocess_start(self._threads, worker, [], daemon=True)
        print()
        for _, t in enumerate(range(self.times)):
            for _, z in enumerate(range(self.slices)):
                printclear(n=1)
                print("t = %d/%d , z = %d/%d" % (t + 1, self.times, z + 1, self.slices))

                Donuts = []
                for cell in cells:
                    if cell.label not in self.list_of_cells:
                        continue
                    if t not in cell.times:
                        continue
                    ti = cell.times.index(t)
                    if z not in cell.zs[ti]:
                        continue
                    zi = cell.zs[ti].index(z)
                    Donuts.append(self._get_donut(cell.label))

                # if self._threads is None:
                results = []
                for donuts in Donuts:
                    cell = self._get_cell(cells, label=donuts.cell_label)
                    ti = cell.times.index(t)
                    zi = cell.zs[ti].index(z)
                    mask_emb = EmbSeg.Embmask[t][z]
                    results.append(
                        donuts.correct_donut_embryo_overlap_c(
                            ti, zi, mask_emb, donuts.cell_label
                        )
                    )

                # else:
                #     TASKS = []
                #     for donuts in Donuts:
                #         cell = self._get_cell(cells, label=donuts.cell_label)
                #         ti = cell.times.index(t)
                #         zi = cell.zs[ti].index(z)
                #         mask_emb = EmbSeg.Embmask[t][z]

                #         TASKS.append((donuts.correct_donut_embryo_overlap_c, (ti, zi, mask_emb, donuts.cell_label)))
                #     task_queue = multiprocess_add_tasks(task_queue, TASKS)
                #     results = multiprocess_get_results(done_queue, TASKS)

                for res in results:
                    donut = self._get_donut(res[0])
                    cell = self._get_cell(cells, res[0])
                    ti = cell.times.index(t)
                    zi = cell.zs[ti].index(z)
                    donut.donut_outlines_out[ti][zi] = res[1]
                    donut.donut_outlines_in[ti][zi] = res[2]
        # if self._threads is not None: multiprocess_end(task_queue)
        printclear()
        return

    def correct_donut_nuclei_overlap(self, cells):
        if self._threads is not None:
            task_queue, done_queue = multiprocess_start(
                self._threads, worker, [], daemon=None
            )

        if self._threads is None:
            for d, donut in enumerate(self.Donuts):
                cell = self._get_cell(cells, donut.cell_label)
                if cell.label not in self.list_of_cells:
                    continue
                donut.compute_donut_masks(cell.masks)
                self.correct_donut_nuclei_overlap_c(donut, cell.masks)
        else:
            TASKS = []
            for d, donut in enumerate(self.Donuts):
                cell = self._get_cell(cells, donut.cell_label)
                args = (
                    cell.label,
                    cell.masks,
                    donut.donut_outlines_out,
                    donut.donut_outlines_in,
                )
                TASKS.append((recompute_donut_masks, args))
            task_queue = multiprocess_add_tasks(task_queue, TASKS)
            results = multiprocess_get_results(done_queue, TASKS)
            for res in results:
                lab = res[0]
                donut = self._get_donut(lab)
                donut.donut_masks = res[1]
                donut.donut_outer_mask = res[2]
                donut.donut_inner_mask = res[3]

            TASKS = []
            for d, donut in enumerate(self.Donuts):
                cell = self._get_cell(cells, self.Donuts[d].cell_label)
                masks = cell.masks
                TASKS.append(
                    (correct_donut_nuclei_overlap_c_paralel, (self.Donuts[d], masks))
                )
            task_queue = multiprocess_add_tasks(task_queue, TASKS)
            results = multiprocess_get_results(done_queue, TASKS)
            self.results = results
            for result in results:
                cell = self._get_cell(cells, result[0])
                donut = self._get_donut(result[0])
                for tid, t in enumerate(cell.times):
                    for zid, z in enumerate(cell.zs[tid]):
                        if result[1][tid][zid] is not None:
                            donut.donut_masks[tid][zid] = result[1][tid][zid]

        if self._threads is not None:
            multiprocess_end(task_queue)
        return

    def correct_donut_nuclei_overlap_c(self, donut, cell_masks):
        for tid in range(len(donut.donut_masks)):
            for zid in range(len(donut.donut_masks[tid])):
                don_mask = donut.donut_masks[tid][zid]
                nuc_mask = cell_masks[tid][zid]
                masks_intersection = intersect2D(don_mask, nuc_mask)
                if len(masks_intersection) == 0:
                    continue
                new_don_mask = get_only_unique(
                    np.vstack((don_mask, masks_intersection))
                )
                donut.donut_masks[tid][zid] = deepcopy(new_don_mask)

    def _get_cell(self, cells, label=None, cellid=None):
        if label == None:
            for cell in cells:
                if cell.id == cellid:
                    return cell
        else:
            for cell in cells:
                if cell.label == label:
                    return cell

        raise Exception("Something is wrong with the inputs of _get_cell")

    def _get_donut(self, label):
        for donuts in self.Donuts:
            if donuts.cell_label == label:
                return donuts

        raise Exception("Something is wrong with the inputs of _get_donut")

    def get_donut_erk(self, cells, img, label, t, z, th=0):
        cell = self._get_cell(cells, label=label)
        tid = cell.times.index(t)
        zid = cell.zs[tid].index(z)

        donuts = self._get_donut(label=label)
        donut = donuts.donut_masks[tid][zid]
        img_cell = np.zeros_like(img)
        xids = donut[:, 1]
        yids = donut[:, 0]
        img_cell[xids, yids] = img[xids, yids]
        erkdonutdist = img[xids, yids]

        nuclei = donuts.nuclei_masks[tid][zid]
        img_cell = np.zeros_like(img)
        xids = nuclei[:, 1]
        yids = nuclei[:, 0]
        img_cell[xids, yids] = img[xids, yids]
        erknucleidist = img[xids, yids]

        erkdonutdist = [x for x in erkdonutdist if x > th]
        erknucleidist = [x for x in erknucleidist if x > th]

        return (
            erkdonutdist,
            erknucleidist,
            np.mean(erkdonutdist) / np.mean(erknucleidist),
        )


def correct_donut_nuclei_overlap_c_paralel(donut, cell_masks):
    new_don_masks = []
    for tid in range(len(donut.donut_masks)):
        new_don_masks.append([])
        for zid in range(len(donut.donut_masks[tid])):
            don_mask = donut.donut_masks[tid][zid]
            nuc_mask = cell_masks[tid][zid]
            masks_intersection = intersect2D(don_mask, nuc_mask)
            if len(masks_intersection) == 0:
                new_don_masks[-1].append(None)
            else:
                new_don_mask = get_only_unique(
                    np.vstack((don_mask, masks_intersection))
                )
                new_don_masks[-1].append(deepcopy(new_don_mask))
    return (donut.cell_label, new_don_masks)
