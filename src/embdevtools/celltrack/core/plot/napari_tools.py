import napari
import vispy.color
import numpy as np

def napari_tracks(cells):
    napari_tracks_data = []
    for cell in cells:
        for tid, t in enumerate(cell.times):
            center = cell.centers[tid]
            track = [cell.label, t, center[0], center[2], center[1]]
            napari_tracks_data.append(track)
    
    return np.array(napari_tracks_data)

def get_lineage_graph(CTB):
    graph = {}
    for mito_ev in CTB.mitotic_events:
        cell0 = mito_ev[0]
        cell1 = mito_ev[1]
        cell2 = mito_ev[2]
        graph[cell1[0]] = [cell0[0]]
        graph[cell2[0]] = [cell0[0]]
    return graph

def arboretum_napari(CTB):


    controls, colors = CTB._plot_args["labels_colors"].get_map()
    custom_cmap = vispy.color.Colormap(colors, controls)

    graph = get_lineage_graph(CTB)

    napari_tracks_data = napari_tracks(CTB.jitcells)
    colors = [CTB._plot_args["labels_colors"].get_control(label) for label in napari_tracks_data[:,0]]
    properties = {'colors': colors}

    viewer = napari.view_image(
        CTB.plot_stacks, 
        name='hyperstack', 
        scale=(CTB.metadata["Zresolution"], 
               CTB.metadata["XYresolution"], 
               CTB.metadata["XYresolution"]), 
        rgb=False, 
        ndisplay=3
    )

    tracks_layer = viewer.add_tracks(
        napari_tracks_data, 
        name="tracks",
        scale=(CTB.metadata["Zresolution"], 
                CTB.metadata["XYresolution"], 
                CTB.metadata["XYresolution"]), 
        properties=properties, 
        graph=graph, 
        color_by='colors',
        colormaps_dict={"colors": custom_cmap}
    )

    points_layer = viewer.add_points(
        napari_tracks_data[:, 1:]*(1, CTB.metadata["Zresolution"], CTB.metadata["XYresolution"], CTB.metadata["XYresolution"]),
        size=2,
        name="centers",
        properties=properties,
        edge_color='colors',
        edge_colormap=custom_cmap,
        face_color='colors',
        face_colormap=custom_cmap,
    )

    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-arboretum", widget_name="Arboretum"
    )
