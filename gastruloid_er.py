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


### PREPROCESSING ###
# Run centroid correction prior to Fijiyama registration to improve performance
IMGS_corrected = centroid_correction_3d_based_on_mid_plane(IMGS)
# Check whether correction is good enough
err = test_mid_plane_centroid_correction(IMGS_corrected, 0, pixel_tolerance=1)


### FJIYAMA REGISTRATION ###
# Create Fijiyama file system (input and output folders)
path_registered, path_output, path_movies_reg = generate_fijiyama_file_system(path_parent, 'movies_registered', embcode)
# Save registration stacks into input folder
generate_fijiyama_stacks(path_registered, IMGS, xyres, zres, file_format="t%d.tif")
# Open Imagej to run fijiyama registration
openfiji()
# Remove stacks used for registration
remove_dir(path_registered)
# Create transformation folders
path_trans_emb_global, path_trans_emb_steps = create_transformations_folders(path_movies_reg, embcode, trans_folder="transformations")
# Move transformations
move_transformation(path_output, path_trans_emb_global, path_trans_emb_steps)
# Remove Fijiyama output folder 
remove_dir(path_output)

### APPLY TRANSFORMATIONS ###
# Define where you have the beanshell class to be called from beanshell
pth_beanshell = "/home/pablo/Desktop/PhD/ImageRegistration/fijiyama/beanshell/bsh-2.0b4.jar"
text_to_write =  "\n".join([pth_beanshell, path_trans_emb_global, path_data, path_movies_reg])
# Save path information in a text file to be open in beanshell.
temporal_file = correct_path(home)+'tmp.txt'
with open(temporal_file, 'w') as the_file:
    the_file.write(text_to_write)

# Run Beanshell script
# subprocess.run(`/opt/Fiji.app/ImageJ-linux64 -batch /home/pablo/Desktop/PhD/ImageRegistration/fijiyama/apply_transform_img/apply_global_img.bsh`)


# Remove path file
os.remove(temporal_file)