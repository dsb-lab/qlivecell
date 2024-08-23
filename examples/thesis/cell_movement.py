### LOAD PACKAGE ###
from qlivecell import cellSegTrack, compute_movement_cell

path_data='/path/to/data/'
path_save='/path/to/save/'

path_data='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/Lineage_2hr_082119_p1/'
path_save='/home/pablo/Desktop/PhD/projects/Data/blastocysts/Claire/2h_claire_ERK-KTR_MKATE2/ctobjects/'

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

if __name__ == "__main__":

    cST = cellSegTrack(
        path_data,
        path_save,
        channels=[1, 0]
    )

    cST.load()

    movements = [compute_movement_cell(cell, "xyz", "center") for cell in cST.jitcells]