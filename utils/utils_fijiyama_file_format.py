from utils.utils_general import create_dir, remove_dir, correct_path
import tifffile
import os
import subprocess
import shutil

def generate_fijiyama_file_system(path, forlder_name, embcode):

    # Create main directory where the final movies will be stored
    create_dir(path, forlder_name)

    # Create a temporal folder where the expanded time-stacks will be stored
    # This folder will be deleted as a whole once the registration is over
    tmp_path= correct_path(correct_path(path)+forlder_name)
    tmp_emb_path = create_dir(tmp_path, 'tmp_'+embcode, rem=True, return_path=True)
    tmp_output_path = create_dir(tmp_path, 'output', rem=True, return_path=True)
    # File system is ready, now files should be stored inside the embcode folder
    return tmp_emb_path, tmp_output_path, tmp_path

def remove_fijiyama_file_system(path, forlder_name=None, embcode=None):

    # Remove temporal folder used for fijiyama registration
    path = correct_path(path)
    if forlder_name is None: remove_dir(path)
    else:
        tmp_path= correct_path(path+forlder_name)
        remove_dir(tmp_path, 'tmp_'+embcode)
        remove_dir(tmp_path, 'output')

def generate_fijiyama_stacks(path_to_save, IMGS, xyres, zres,file_format="t%d.tif", rem=True):
    pth = correct_path(path_to_save)
    ts,zs,xs,ys = IMGS.shape
    list_of_files = os.listdir(pth)
    if len(list_of_files)!=0:
        if rem:
            remove_dir(pth)
            create_dir(pth)
    for t in range(ts):
        IMG = IMGS[t].reshape((1,zs,xs,ys))
        fullpath = pth+file_format %(t+1)
        mdata = {'axes': 'TZYX', 'spacing': zres, 'unit': 'um'}
        tifffile.imwrite(fullpath, IMG, imagej=True, resolution=(1/xyres, 1/xyres), metadata=mdata)

def openfiji(path_to_fiji='/opt/Fiji.app/ImageJ-linux64'):
    subprocess.run([path_to_fiji], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def create_transformations_folders(path_movies, embcode, trans_folder="transformations"):
    path_transformations = create_dir(path_movies, trans_folder, return_path=True)
    path_trans_embcode   = create_dir(path_transformations, embcode, return_path=True)
    path_trans_emb_global= create_dir(path_trans_embcode, 'global', return_path=True)
    path_trans_emb_steps = create_dir(path_trans_embcode, 'steps', return_path=True)

    return path_trans_emb_global, path_trans_emb_steps

def move_transformation(path_output, path_trans_emb_global, path_trans_emb_steps):
    pth_to_output_exported_data = correct_path(path_output)+ 'Exported_data'
    try: 
        ex_data_files = os.listdir(pth_to_output_exported_data)
        for file in ex_data_files:
            if 'global' in file:
                file_path = correct_path(pth_to_output_exported_data)+file
                new_file_path = correct_path(path_trans_emb_global) + file
                shutil.move(file_path, new_file_path)
    except: pass
    pth_to_output_registration_files = correct_path(path_output)+ 'Registration_files'
    try: 
        rf_data_files = os.listdir(pth_to_output_registration_files)
        for file in rf_data_files:
            if 'Transform_Step' in file:
                file_path = correct_path(pth_to_output_registration_files)+file
                new_file_path = correct_path(path_trans_emb_steps) + file
                shutil.move(file_path, new_file_path)
    except: pass
