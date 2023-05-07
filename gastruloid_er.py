from embryoregistration import *
import os

home = os.path.expanduser('~')
path_parent = home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/'
path_data=path_parent+'movies/'
path_save=path_parent+'CellTrackObjects/'

file, embcode = get_file_embcode(path_data, 0)

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

IMGS_corrected = IMGS_corrected.astype('uint8')

### FJIYAMA REGISTRATION ###
# Create Fijiyama file system (input and output folders)
path_registered, path_output, path_movies_reg = generate_fijiyama_file_system(path_parent, 'movies_registered', embcode)

# Save registration stacks into input folder
generate_fijiyama_stacks(path_registered, IMGS_corrected, xyres, zres, file_format="t%d.tif")

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

path_movies_reg_embcode = create_dir(path_movies_reg, embcode, return_path=True, rem=True)
IMGS_chs = np.array([IMGS_ch0,IMGS_ch1,IMGS_ch2])

# Expand channels
registered_IMGS_chs = np.zeros_like(np.array(IMGS_chs))

for ch, IMGS_ch in enumerate(IMGS_chs):

    path_movies_reg_embcode_ch = create_dir(path_movies_reg_embcode,'%d' %ch, return_path=True, rem=True)
    path_movies_reg_embcode_ch_reg = create_dir(path_movies_reg_embcode,'registered_%d' %ch, return_path=True, rem=True)

    generate_fijiyama_stacks(path_movies_reg_embcode_ch, IMGS_ch, xyres, zres, file_format="t%d.tif", rem=True)

    # Define where you have the beanshell class to be called from beanshell
    pth_beanshell = "/opt/Fiji.app/beanshell/bsh-2.0b4.jar"
    text_to_write =  "\n".join([pth_beanshell, path_trans_emb_global, path_movies_reg_embcode_ch, correct_path(path_movies_reg_embcode_ch_reg)])
    # Save path information in a text file to be open in beanshell.
    temporal_file = correct_path(home)+'tmp.txt'
    with open(temporal_file, 'w') as the_file:
        the_file.write(text_to_write)
    
    # Run Beanshell script
    pth_beanshell_script = correct_path(os.getcwd())+'utils/apply_transformation.bsh'
    subprocess.run(['/opt/Fiji.app/ImageJ-linux64', '--headless' ,'--run', pth_beanshell_script])#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    remove_dir(path_movies_reg_embcode_ch)
    # Remove path file
    os.remove(temporal_file)
    
    tfiles = os.listdir(path_movies_reg_embcode_ch_reg)
    for _t, tfile in enumerate(tfiles):
        registered_IMGS_chs[ch]
        t=int(tfile.split('.')[0][-1])-1
        IMG_t, xyres, zres = read_img_with_resolution(correct_path(path_movies_reg_embcode_ch_reg)+tfile, channel=None)
        registered_IMGS_chs[ch][t]=IMG_t

# Reorder stack from CTZXY to the imagej structure TZCXY 
sh = registered_IMGS_chs.shape
final_registered_IMGS_chs = np.zeros((sh[1], sh[2], sh[0], sh[3], sh[4]))
for ch in range(sh[0]):
    for t in range(sh[1]):
        for z in range(sh[2]):
            final_registered_IMGS_chs[t,z,ch] = registered_IMGS_chs[ch, t, z]

# Convert again to 8 bit
final_registered_IMGS_chs = final_registered_IMGS_chs.astype('uint8')

# Save final stack and remove unncessary intermediate files
fullpath = path_movies_reg_embcode+'.tif'
mdata = {'axes': 'TZCYX', 'spacing': zres, 'unit': 'um'}
tifffile.imwrite(fullpath, final_registered_IMGS_chs, imagej=True, resolution=(1/xyres, 1/xyres), metadata=mdata)
remove_dir(path_movies_reg_embcode)
