import napari
import numpy as np
import vispy.color


def napari_tracks(cells):
    napari_tracks_data = []
    for cell in cells:
        for tid, t in enumerate(cell.times):
            center = cell.centers[tid]
            track = [cell.label, t, center[0], center[2], center[1]]
            napari_tracks_data.append(track)

    return np.array(napari_tracks_data)


def get_lineage_graph(mitotic_events):
    graph = {}
    for mito_ev in mitotic_events:
        cell0 = mito_ev[0]
        cell1 = mito_ev[1]
        cell2 = mito_ev[2]
        graph[cell1[0]] = cell0[0]
        graph[cell2[0]] = cell0[0]

    return graph


def get_lineage_root(graph, label):
    # Check if this label is the root of the lineage
    root = False
    lab = label
    # If it's in the keys, this is not the root so look for the root
    while not root:
        if lab in list(graph.keys()):
            lab = graph[lab]
        else:
            root = True
    return lab


def get_mothers(graph, label):
    # Check if this label is the root of the lineage
    root = False
    lab = label
    labels = []

    # If it's in the keys, this is not the root so look for the root
    while not root:
        if lab in list(graph.keys()):
            lab = graph[lab]
            labels.append(lab)
        else:
            root = True
    return labels


def get_lineage(graph, label):
    labels = [label]
    # Check if this label is the root of the lineage
    root = False
    lab = label
    # If it's in the keys, this is not the root so look for the root
    while not root:
        if lab in list(graph.keys()):
            lab = graph[lab]
            labels.append(lab)
        else:
            root = True
    return labels


def get_all_lineages(mitotic_events):
    roots = []


def get_all_daughters(labels, graph, lab):
    vals = np.array(list(graph.values()))
    keys = np.array(list(graph.keys()))

    idxs = np.where(vals == lab)[0]
    last = False
    while not last:
        if len(idxs) == 0:
            last = True
        else:
            for idx in idxs:
                labels.append(keys[idx])
                get_all_daughters(labels, graph, labels[-1])
            last = True
    return labels


def get_daughters(graph, lab):
    vals = np.array(list(graph.values()))
    keys = np.array(list(graph.keys()))
    labels = []
    idxs = np.where(vals == lab)[0]
    last = False
    while not last:
        for idx in idxs:
            labels.append(keys[idx])
        last = True
    return labels


def get_lineage_ends(graph, lineage):
    ends = []
    mothers = np.unique(list(graph.values()))
    for lab in lineage:
        if lab not in mothers:
            ends.append(lab)
    return ends


def get_whole_lineage(mitotic_events, label):
    graph = get_lineage_graph(mitotic_events)
    root = get_lineage_root(graph, label)
    labels = [root]
    lab = root
    labels = get_all_daughters(labels, graph, lab)
    return labels


def arboretum_napari(cellSegTrack_instance):
    cST = cellSegTrack_instance
    controls, colors = cST._plot_args["labels_colors"].get_map()
    custom_cmap = vispy.color.Colormap(colors, controls)

    graph = get_lineage_graph(cST.mitotic_events)

    napari_tracks_data = napari_tracks(cST.jitcells)
    colors = [
        cST._plot_args["labels_colors"].get_control(label)
        for label in napari_tracks_data[:, 0]
    ]
    properties = {"colors": colors}

    viewer = napari.view_image(
        cST.hyperstack[:, :, cST.channels_order[0]],
        name="hyperstack",
        scale=(
            cST.metadata["Zresolution"],
            cST.metadata["XYresolution"],
            cST.metadata["XYresolution"],
        ),
        rgb=False,
        ndisplay=3,
    )

    tracks_layer = viewer.add_tracks(
        napari_tracks_data,
        name="tracks",
        scale=(
            cST.metadata["Zresolution"],
            cST.metadata["XYresolution"],
            cST.metadata["XYresolution"],
        ),
        properties=properties,
        graph=graph,
        color_by="colors",
        colormaps_dict={"colors": custom_cmap},
    )

    points_layer = viewer.add_points(
        napari_tracks_data[:, 1:]
        * (
            1,
            cST.metadata["Zresolution"],
            cST.metadata["XYresolution"],
            cST.metadata["XYresolution"],
        ),
        size=2,
        name="centers",
        properties=properties,
        edge_color="colors",
        edge_colormap=custom_cmap,
        face_color="colors",
        face_colormap=custom_cmap,
    )

    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-arboretum", widget_name="Arboretum"
    )
