import numpy as np
from munkres import Munkres
from scipy.spatial.distance import directed_hausdorff

from .utils_ct import printfancy


def greedy_tracking(TLabels, TCenters, xyresolution, zresolution, track_args):
    dist_th = track_args["dist_th"]
    z_th = track_args["z_th"]

    z_th_units = int(np.rint(z_th / zresolution))
    FinalLabels = []
    FinalCenters = []
    label_correspondance = []
    print(TLabels)
    # for each time track to the previous one
    for t in range(len(TLabels)):
        label_correspondance.append([])
        # if the first time, labels remain the same
        if t == 0:
            FinalLabels.append(TLabels[0])
            FinalCenters.append(TCenters[0])
            labmax = np.max(FinalLabels[0])
            for lab in TLabels[0]:
                label_correspondance[0].append([lab, lab])
            continue

        # if not first time, we need to fill the correspondance
        FinalLabels.append([])
        FinalCenters.append([])
        # pre-allocate distance matrix of shape [labs at t-1, labs at t]
        Dists = np.ones((len(FinalLabels[t - 1]), len(TLabels[t])))

        # for each label at t-1
        for i in range(len(FinalLabels[t - 1])):
            # position of ith cell at t-1
            poscell1 = np.array(FinalCenters[t - 1][i][1:]) * np.array(
                [xyresolution, xyresolution]
            )

            # for each cell at t
            for j in range(len(TLabels[t])):
                # position of jth cell at t
                poscell2 = np.array(TCenters[t][j][1:]) * np.array(
                    [xyresolution, xyresolution]
                )

                # compute distance between the two
                Dists[i, j] = np.linalg.norm(poscell1 - poscell2)

                # check if cell cell centers are separated by more than z_th slices
                zdisp = np.abs(FinalCenters[t - 1][i][0] - TCenters[t][j][0])
                zdisp_units = int(np.rint(zdisp / zresolution))

                if zdisp_units > z_th_units:
                    # if so, set the distance to a large number (e.g. 100)
                    Dists[i, j] = 100.0

        # for each future cell, which is their closest past one
        a = np.argmin(Dists, axis=0)  # max prob for each future cell to be a past cell

        # for each past cell, which is their closest future one
        b = np.argmin(Dists, axis=1)  # max prob for each past cell to be a future one

        correspondance = []
        notcorrespondenta = []
        notcorrespondentb = []

        # for each past cell
        for i, j in enumerate(b):
            # j is the index of the closest future cell to cell i
            # check if the closes cell to j cell is i
            if i == a[j]:
                # check if their distance is below a th
                if Dists[i, j] < dist_th:
                    # save correspondance and final label
                    correspondance.append([i, j])  # [past, future]
                    label_correspondance[t].append(
                        [TLabels[t][j], FinalLabels[t - 1][i]]
                    )
                    FinalLabels[t].append(FinalLabels[t - 1][i])
                    FinalCenters[t].append(TCenters[t][j])

            else:
                # if there was no correspondance, save that
                notcorrespondenta.append(i)

        # update max label
        labmax = np.maximum(np.max(FinalLabels[t - 1]), labmax)

        # for each future cell
        for j in range(len(a)):
            # check if the future cell is in the correspondance
            if j not in np.array(correspondance)[:, 1]:
                # if not, save it as a new label
                label_correspondance[t].append([TLabels[t][j], labmax + 1])
                FinalLabels[t].append(labmax + 1)
                FinalCenters[t].append(TCenters[t][j])
                labmax += 1
                notcorrespondentb.append(j)

    return FinalLabels, label_correspondance


def hungarian_tracking(
    TLabels, TCenters, TOutlines, TMasks, xyresolution, zresolution, track_args
):
    z_th = track_args["z_th"]
    z_th_units = int(np.rint(z_th / zresolution))

    cost_attributes = track_args["cost_attributes"]
    cost_ratios = track_args["cost_ratios"]

    cost_dict = dict(zip(cost_attributes, cost_ratios))

    FinalLabels = []
    label_correspondance = []

    FinalLabels.append(TLabels[0])
    lc = [[l, l] for l in TLabels[0]]
    label_correspondance.append(lc)

    labmax = 0
    for t in range(1, len(TLabels)):
        FinalLabels_t = []
        label_correspondance_t = []

        labs1 = TLabels[t - 1]
        labs2 = TLabels[t]

        pos1 = TCenters[t - 1]
        pos2 = TCenters[t]

        masks1 = TMasks[t - 1]
        masks2 = TMasks[t]

        outs1 = TOutlines[t - 1]
        outs2 = TOutlines[t]

        cost_matrix = []
        for i in range(len(labs1)):
            row = []
            for j in range(len(labs2)):
                zdisp = np.abs(pos1[i][0] - pos2[j][0])
                zdisp_units = int(np.rint(zdisp / zresolution))

                if zdisp_units > z_th_units:
                    distance = 100.0
                else:
                    distance = (
                        (pos1[i][1] - pos2[j][1]) ** 2 + (pos1[i][2] - pos2[j][2]) ** 2
                    ) ** 0.5
                    distance *= xyresolution

                vol1 = len(masks1[i])
                vol2 = len(masks2[j])
                volume_diff = abs(vol1 - vol2)
                shape_diff = directed_hausdorff(outs1[i], outs2[j])[
                    0
                ]  # Hausdorff distance
                cost = 0
                if "distance" in cost_attributes:
                    cost += distance * cost_dict["distance"]
                if "volume" in cost_attributes:
                    cost += volume_diff * cost_dict["volume"]
                if "shape" in cost_attributes:
                    cost += shape_diff * cost_dict["shape"]
                row.append(cost)

            cost_matrix.append(row)

        # Create an instance of the Munkres class
        m = Munkres()

        # Solve the assignment problem using the Hungarian algorithm
        try:
            indexes = m.compute(cost_matrix)
        except IndexError:
            indexes = []

        # Print the matched cell pairs
        for row, column in indexes:
            label1 = labs1[row]
            # get the updated corresponding label
            label1idx = np.where(np.array(label_correspondance[t - 1])[:, 0] == label1)[
                0
            ][0]

            label1 = np.array(label_correspondance[t - 1])[:, 1][label1idx]

            label2 = labs2[column]
            label_correspondance_t.append([label2, label1])
            FinalLabels_t.append(label1)

        if len(FinalLabels[t - 1]) != 0:
            labmax = np.maximum(np.max(FinalLabels[t - 1]), labmax)
        for lab in labs2:
            if lab not in np.array(label_correspondance_t)[:, 0]:
                labmax += 1
                label_correspondance_t.append([lab, labmax])
                FinalLabels_t.append(labmax)
        FinalLabels.append(FinalLabels_t)
        label_correspondance.append(label_correspondance_t)

    return FinalLabels, label_correspondance


"""
checks necessary arguments.
"""


def check_tracking_args(tracking_arguments, available_tracking=["greedy", "hungarian"]):
    if "method" not in tracking_arguments.keys():
        printfancy("No tracking method provided. Using greedy algorithm")
        printfancy()
        tracking_arguments["method"] = "greedy"

    if "time_step" not in tracking_arguments.keys():
        printfancy("No time step provided, using 1 minute.")
        printfancy()
        tracking_arguments["time_step"] = 1

    if tracking_arguments["method"] not in available_tracking:
        raise Exception("invalid segmentation method")
    return


"""
fills rest of arguments
"""


def fill_tracking_args(tracking_arguments):
    tracking_method = tracking_arguments["method"]

    if tracking_method == "hungarian":
        new_tracking_arguments = {
            "time_step": 1,
            "method": "hungarian",
            "z_th": 2,
            "cost_attributes": ["distance", "volume", "shape"],
            "cost_ratios": [0.6, 0.2, 0.2],
        }

    elif tracking_method == "greedy":
        new_tracking_arguments = {
            "time_step": 1,
            "method": "greedy",
            "dist_th": 7.5,
            "z_th": 2,
        }

    for targ in tracking_arguments.keys():
        try:
            new_tracking_arguments[targ] = tracking_arguments[targ]
        except KeyError:
            raise Exception(
                "key %s is not a correct argument for the selected tracking method"
                % targ
            )

    return new_tracking_arguments
