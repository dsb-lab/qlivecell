from tifffile import TiffFile, imwrite
import numpy as np


img = np.random.random((4, 10, 20,20))*255
img = np.rint(img).astype("uint8")
imagejformat = "TZYX"
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

with TiffFile("test.tif") as tif:
    preIMGS = tif.asarray()
    imagej_metadata = tif.imagej_metadata
    tags = tif.pages[0].tags

imagej_metadata