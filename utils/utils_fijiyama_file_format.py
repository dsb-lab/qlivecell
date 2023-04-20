from utils.utils_general import create_dir, remove_dir, correct_path
import tifffile
import os
import subprocess

def generate_fijiyama_file_system(path, forlder_name, embcode):

    # Create main directory where the final movies will be stored
    create_dir(path, forlder_name)

    # Create a temporal folder where the expanded time-stacks will be stored
    # This folder will be deleted as a whole once the registration is over
    tmp_path= correct_path(correct_path(path)+forlder_name)
    create_dir(tmp_path, 'tmp_'+embcode)

    # File system is ready, now files should be stored inside the embcode folder
    return correct_path(tmp_path+ 'tmp_'+embcode)

def remove_fijiyama_file_system(path, forlder_name=None, embcode=None):

    # Remove temporal folder used for fijiyama registration
    path = correct_path(path)
    if forlder_name is None: remove_dir(path)
    else:
        tmp_path= correct_path(path+forlder_name)
        remove_dir(tmp_path, 'tmp_'+embcode)


def generate_fijiyama_stacks(path_to_save, IMGS, xyres, zres,file_format="t_%d.tif"):
    pth = correct_path(path_to_save)
    ts,zs,xs,ys = IMGS.shape
    list_of_files = os.listdir(pth)
    if len(list_of_files)!=0:
        remove_dir(pth)
        create_dir(pth)
    for t in range(ts):
        IMG = IMGS[t] 
        fullpath = pth+file_format %(t+1)
        mdata = {'spacing': zres, 'unit': 'um'}
        tifffile.imwrite(fullpath, IMG, imagej=True, resolution=(xyres, xyres), metadata=mdata)

def save_stack_for_fijiyama():
    pass

def openfiji(path_to_fiji='/opt/Fiji.app/ImageJ-linux64'):
    subprocess.run([path_to_fiji], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
