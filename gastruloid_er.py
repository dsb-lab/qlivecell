from embryoregistration import *
import os

home = os.path.expanduser('~')
path_parent = home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/'
path_data=path_parent+'movies/'
path_save=path_parent+'CellTrackObjects/'

file, embcode = get_file_embcode(path_data, 1)

# Extract both channels
IMGS_ch0, xyres, zres = read_img_with_resolution(path_data+file, channel=0)
IMGS_ch1, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS_ch2, xyres, zres = read_img_with_resolution(path_data+file, channel=2)

# Combine channels into single stack
IMGS = IMGS_ch0 + IMGS_ch1 #+ IMGS_ch2
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(IMGS_ch1[0][-1])
plt.show()

IMGS_corrected = centroid_correction_3d_based_on_mid_plane(IMGS)
err = test_mid_plane_centroid_correction(IMGS_corrected, 0, pixel_tolerance=1)

assert err[1]

path_registered = generate_fijiyama_file_system(path_parent, 'movies_registered', embcode)
generate_fijiyama_stacks(path_registered, IMGS, xyres, zres, file_format="t_%d.tif")

openfiji()

remove_fijiyama_file_system(path_registered)