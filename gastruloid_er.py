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
IMGS = IMGS_ch0.astype('uint16') + IMGS_ch1.astype('uint16') #+ IMGS_ch2.astype('uint16')
t, z, x, y = np.where(IMGS>255)
IMGS[t,z,x,y] = 255
IMGS = IMGS.astype('uint8')

IMGS_corrected = centroid_correction_3d_based_on_mid_plane(IMGS)
err = test_mid_plane_centroid_correction(IMGS_corrected, 0, pixel_tolerance=1)

assert err[1]

path_registered = generate_fijiyama_file_system(path_parent, 'movies_registered', embcode)
generate_fijiyama_stacks(path_registered, IMGS_corrected, xyres, zres, file_format="t_%d.tif")

openfiji()

remove_fijiyama_file_system(path_registered)
