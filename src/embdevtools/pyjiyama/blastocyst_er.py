import sys

sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')

import os

from embdevtools.embdevtools import (embryoregistration, get_file_embcode,
                         read_img_with_resolution)

home = os.path.expanduser("~")
path_parent = (
    home + '/'
)
path_data = path_parent + "volumes/volumes/"

import numpy as np

file, embcode, files = get_file_embcode(path_data, "combined", returnfiles=True)
_IMGS, xyres, zres = read_img_with_resolution(path_data + file)

IMGS = embryoregistration.square_stack4D(_IMGS[:91])
del _IMGS

# for i in range(1, 10):
#     file, embcode, files = get_file_embcode(path_data, "00%d.tif" % i, returnfiles=True)

#     _IMGS, xyres, zres = read_img_with_resolution(path_data + file)
#     _IMGS = _IMGS[0:2, :, 1:-2, :]
#     IMGS = np.append(IMGS, _IMGS, axis=0)

# Combine channels into single stack
t, z, x, y = np.where(IMGS > 255)
IMGS[t, z, x, y] = 255
IMGS = IMGS.astype("uint8")

### PREPROCESSING ###
# Run centroid correction prior to Fijiyama registration to improve performance
IMGS_corrected = embryoregistration.centroid_correction_3d_based_on_mid_plane(IMGS)
del IMGS
# Check whether correction is good enough
# err = test_mid_plane_centroid_correction(IMGS_corrected, 0, pixel_tolerance=1)

IMGS_corrected = IMGS_corrected.astype("uint8")

### FJIYAMA REGISTRATION ###
# Create Fijiyama file system (input and output folders)
(
    path_registered,
    path_output,
    path_movies_reg,
) = embryoregistration.generate_fijiyama_file_system(
    path_data, "movies_registered", embcode
)

# Save registration stacks into input folder
embryoregistration.generate_fijiyama_stacks(
    path_registered, IMGS_corrected, xyres, zres, file_format="t%d.tif"
)

# Open Imagej to run fijiyama registration
embryoregistration.openfiji()

# Remove stacks used for registration
embryoregistration.remove_dir(path_registered)

# Create transformation folders
(
    path_trans_emb_global,
    path_trans_emb_steps,
) = embryoregistration.create_transformations_folders(
    path_movies_reg, embcode, trans_folder="transformations"
)

# Move transformations
embryoregistration.move_transformation(
    path_output, path_trans_emb_global, path_trans_emb_steps
)

# Remove Fijiyama output folder
embryoregistration.remove_dir(path_output)

### APPLY TRANSFORMATIONS ###

path_movies_reg_embcode = embryoregistration.create_dir(
    path_movies_reg, embcode, return_path=True, rem=True
)
IMGS_chs = np.array([IMGS_corrected])

# Expand channels
registered_IMGS_chs = np.zeros_like(np.array(IMGS_chs))

for ch, IMGS_ch in enumerate(IMGS_chs):
    path_movies_reg_embcode_ch = embryoregistration.create_dir(
        path_movies_reg_embcode, "%d" % ch, return_path=True, rem=True
    )
    path_movies_reg_embcode_ch_reg = embryoregistration.create_dir(
        path_movies_reg_embcode, "registered_%d" % ch, return_path=True, rem=True
    )

    embryoregistration.generate_fijiyama_stacks(
        path_movies_reg_embcode_ch,
        IMGS_ch,
        xyres,
        zres,
        file_format="t%d.tif",
        rem=True,
    )

    # Define where you have the beanshell class to be called from beanshell
    pth_beanshell = "/opt/Fiji.app/beanshell/bsh-2.0b4.jar"
    text_to_write = "\n".join(
        [
            pth_beanshell,
            path_trans_emb_global,
            path_movies_reg_embcode_ch,
            embryoregistration.correct_path(path_movies_reg_embcode_ch_reg),
        ]
    )
    # Save path information in a text file to be open in beanshell.
    temporal_file = embryoregistration.correct_path(home) + "tmp.txt"
    with open(temporal_file, "w") as the_file:
        the_file.write(text_to_write)

    # Run Beanshell script
    pth_beanshell_script = (
        embryoregistration.correct_path(
            "/home/pablo/Desktop/PhD/projects/embdevtools/src/embdevtools/pyjiyama/"
        )
        + "utils/apply_transformation.bsh"
    )
    import subprocess

    subprocess.run(
        ["/opt/Fiji.app/ImageJ-linux64", "--headless", pth_beanshell_script]
    )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    embryoregistration.remove_dir(path_movies_reg_embcode_ch)
    # Remove path file
    os.remove(temporal_file)

    tfiles = os.listdir(path_movies_reg_embcode_ch_reg)
    for _t, tfile in enumerate(tfiles):
        registered_IMGS_chs[ch]
        t = int(tfile.split(".")[0][-1]) - 1
        print(t)
        IMG_t, xyres, zres = read_img_with_resolution(
            embryoregistration.correct_path(path_movies_reg_embcode_ch_reg) + tfile,
            channel=None,
        )
        registered_IMGS_chs[ch][t] = IMG_t

# Reorder stack from CTZXY to the imagej structure TZCXY
sh = registered_IMGS_chs.shape
final_registered_IMGS_chs = np.zeros((sh[1], sh[2], sh[0], sh[3], sh[4]))
for ch in range(sh[0]):
    for t in range(sh[1]):
        for z in range(sh[2]):
            final_registered_IMGS_chs[t, z, ch] = registered_IMGS_chs[ch, t, z]

# Convert again to 8 bit
final_registered_IMGS_chs = final_registered_IMGS_chs.astype("uint8")

# Save final stack and remove unncessary intermediate files
fullpath = path_movies_reg_embcode + ".tif"
mdata = {"axes": "TZCYX", "spacing": zres, "unit": "um"}
import tifffile

tifffile.imwrite(
    fullpath,
    final_registered_IMGS_chs,
    imagej=True,
    resolution=(1 / xyres, 1 / xyres),
    metadata=mdata,
)
embryoregistration.remove_dir(path_movies_reg_embcode)
