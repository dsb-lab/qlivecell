from cellpose.io import imread
from cellpose import models
import os
from CellTracking import *
from utils_ct import *
pth='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/registered/'
pthtosave='/home/pablo/Desktop/PhD/projects/Data/blastocysts/CellTrackObjects/2h_claire_ERK-KTR_MKATE2/'

files = os.listdir(pth)
emb = 9
embcode=files[emb][0:-4]
IMGS   = [imread(pth+f)[0:6,:,1,:,:] for f in files[emb:emb+1]][0]
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
#model  = models.Cellpose(gpu=True, model_type='nuclei')

from tifffile import imwrite
CT = CellTracking( IMGS, model, embcode
                    , trainedmodel=True
                    , channels=[0,0]
                    , flow_th_cellpose=0.4
                    , distance_th_z=3.0
                    , xyresolution=0.2767553
                    , relative_overlap=False
                    , use_full_matrix_to_compute_overlap=True
                    , z_neighborhood=2
                    , overlap_gradient_th=0.15
                    , plot_layout=(2,2)
                    , plot_overlap=1
                    , masks_cmap='tab10'
                    , min_outline_length=400
                    , neighbors_for_sequence_sorting=7
                    , plot_tracking_windows=1
                    , backup_steps=5
                    , time_step=5
                    , cell_distance_axis="xy"
                    , mean_substraction_cell_movement=True)

CT()
save_CT(CT, pthtosave, embcode)
#CTl = load_CT(pthtosave, embcode)
CT.stack_dims
masks = np.zeros((CT.times, CT.slices,3, CT.stack_dims[0], CT.stack_dims[1])).astype('float32')

for cell in CT.cells:
    color = np.array(np.array(CT._label_colors[CT._labels_color_id[cell.label]])*255).astype('float32')
    for tid, tc in enumerate(cell.times):
        for zid, zc in enumerate(cell.zs[tid]):
            mask = cell.masks[tid][zid]
            xids = mask[:,1]
            yids = mask[:,0]
            masks[tc][zc][0][xids,yids]=color[0]
            masks[tc][zc][1][xids,yids]=color[1]
            masks[tc][zc][2][xids,yids]=color[2]
#masks = np.reshape(masks, (CT.times, CT.slices, 3, CT.stack_dims[0], CT.stack_dims[1]))

imwrite(
     '/home/pablo/Desktop/temp.tiff',
     masks,
     imagej=True,
     resolution=(1/0.2767553, 1/0.2767553),
     photometric='rgb',
     metadata={
         'spacing': 2.0,
         'unit': 'um',
         #'finterval': 5,
         'axes': 'TZCYX',
     }
)

for i in range(10):
    color = np.array(np.array(CT._label_colors[CT._labels_color_id[i]])*255, dtype='uint8')
    print(color)
plt.imshow(masks[0][0])
plt.show()