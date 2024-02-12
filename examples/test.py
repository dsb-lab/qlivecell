from tifffile import TiffFile, imwrite
import numpy as np


img = np.random.random((4,10,2, 20,20))*255
img = np.rint(img).astype("uint8")
imagejformat = "TZCYX"
imwrite(
    "test.tif",
    img,
    imagej=True,
    resolution=(5, 5),
    resolutionunit=5,
    metadata={
        "spacing": 1,
        "unit": "um",
        "finterval": 300,
        "axes": imagejformat,
    },
)

def tif_reader_5D(path_to_file):
    """
    Parameters
    ----------
    path_to_file : str
        The path to the tif file.

    Returns
    -------
    hyperstack: 
        5D numpy array with shape (t, z, c, x, y)
    metadata:
        Dict containing imagej metadata and xy and z spacings (inverse of resolution)
        
    """
    with TiffFile(path_to_file) as tif:
        hyperstack = tif.asarray()
        imagej_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags
        
        metadata_keys = imagej_metadata.keys()
        
        try: 
            frames = imagej_metadata['frames']
        except KeyError:
            frames = 1
        
        try:
            slices = imagej_metadata['slices']
        except KeyError:
            slices = 1
        
        try: 
            channels = imagej_metadata['channels']
        except KeyError:
            channels=1

        hyperstack = np.reshape(hyperstack, (frames, slices, channels, *hyperstack.shape[-2:]))

        # parse X, Y resolution
        try:
            npix, unit = tags["XResolution"].value
            xres = unit / npix
        except KeyError:
            xres = 1

        try:
            npix, unit = tags["YResolution"].value
            yres = unit / npix
        except KeyError:
            yres = 1


        try:
            res_unit= tags["ResolutionUnit"].value
        except KeyError:
            yres = 1

        try:
            zres = imagej_metadata["spacing"]
        except:
            zres = 1

        if xres == yres:
            xyres = xres
        else:
            xyres = np.mean([xres, yres])
    
    imagej_metadata["xyres"] = xyres
    imagej_metadata["zres"]  = zres
    imagej_metadata["res_unit"] = res_unit
    return hyperstack, imagej_metadata

hyperstack, metadata = tif_reader_5D("test.tif")


### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

embcode = '20230607_CAG_H2B_GFP_16_cells_stack2_registered'
path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Lana/20230607_CAG_H2B_GFP_16_cells/stack_2_channel_0_obj_bottom/crop/'+embcode

hyperstack, metadata = tif_reader_5D(path_data+"/0.tif")
