import numpy as np
from .utils_ct import get_default_args

def train_CellposeModel(train_imgs, train_masks, model, train_seg_args):

    from cellpose import models

    model = models.CellposeModel(
        gpu=model.gpu, pretrained_model=model.pretrained_model[0]
    )
    modelpath = model.train(
        train_imgs,
        train_masks,
        **train_seg_args
    )
    model = models.CellposeModel(gpu=model.gpu, pretrained_model=modelpath)
    return model


def train_StardistModel(train_imgs, train_masks, model, train_seg_args):
    
    model.train(
        train_imgs,
        train_masks,
        **train_seg_args
    )
    return model



def blur_imgs(img, blur_args):
    blur_img = img.copy()
    if len(img.shape)==3:
        for ch in range(img.shape[-1]):
            blur_img[:,:,ch] = GaussianBlur(img[:,:,ch], blur_args[0], blur_args[1])
    else: 
        blur_img[:,:,ch] = GaussianBlur(img, blur_args[0], blur_args[1])
    return blur_img


from cv2 import GaussianBlur


def get_training_set(IMGS, Masks_stack, tz_actions, train_args):
    blur_args = train_args["blur"]

    tzt = np.asarray(tz_actions)
    tzt = np.unique(tzt, axis=0)
    train_imgs = []
    train_masks = []

    for act in tzt:
        t, z = act
        img = IMGS[t, z]
        msk = Masks_stack[t, z]
        if blur_args is not None:
            img = blur_imgs(img, blur_args)
        train_imgs.append(img)
        train_masks.append(msk)

    return train_imgs, train_masks

from datetime import datetime


def check_and_fill_train_segmentation_args(train_segmentation_args, model, seg_method, path_to_save):
    
    if model is None:
        return None, None
    else:
        train_seg_args = get_default_args(model.train)
    
    new_train_segmentation_args = {
            "blur": None,
        }

    if "blur" not in train_segmentation_args.keys():
        new_train_segmentation_args["blur"] = None
    
    # Define model save dir and name for all segmentation methods
    if 'cellpose' in seg_method:
        path_save_arg = "save_path"
        model_name_arg = "model_name"
        
        if path_save_arg not in train_segmentation_args.keys():
            train_segmentation_args[path_save_arg] = path_to_save
    
        if model_name_arg not in train_segmentation_args.keys() or train_segmentation_args[model_name_arg] is None: 
            now = datetime.now()
            dt = now.strftime(seg_method + "_%d-%m-%Y_%H-%M-%S")
            train_segmentation_args[model_name_arg] = dt
            
        config_args = {}
    
    elif 'stardist' in seg_method:
        path_save_arg = "base_fir"
        model_name_arg = "name"
        
        if path_save_arg not in train_segmentation_args.keys():
            train_segmentation_args[path_save_arg] = path_to_save

        if model_name_arg not in train_segmentation_args.keys() or train_segmentation_args[model_name_arg] is None:
            now = datetime.now()
            dt = now.strftime(seg_method + "_%d-%m-%Y_%H-%M-%S")
            train_segmentation_args[model_name_arg] = dt
    
        model.base_dir = train_segmentation_args.pop(path_save_arg)
        model.name = train_segmentation_args.pop(model_name_arg)
        
        config_args = model.config.__dict__

    for tsarg in train_segmentation_args.keys():
        if tsarg in new_train_segmentation_args.keys():
             new_train_segmentation_args[tsarg] = train_segmentation_args[tsarg]
        elif tsarg in train_seg_args.keys():
            train_seg_args[tsarg] = train_segmentation_args[tsarg]
        # In the case of Stardist, most training arguments are part of the model config
        elif tsarg in config_args.keys():
            model.config.__dict__[tsarg]=train_segmentation_args[tsarg]
        else:
            raise Exception(
                "key %s is not a correct training argument for the selected segmentation method"
                % tsarg
            )
            
    return new_train_segmentation_args, train_seg_args
