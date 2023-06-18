import numpy as np

def train_CellposeModel(train_imgs, train_masks, train_seg_args, model, channels):    
    
    path_save = train_seg_args['model_save_path']
    model_name = train_seg_args['model_name']
    
    from cellpose import models
    model  = models.CellposeModel(gpu=model.gpu, pretrained_model=model.pretrained_model[0])
    modelpath = model.train(train_imgs, train_masks, channels = channels, save_path=path_save, model_name=model_name)
    model  = models.CellposeModel(gpu=model.gpu, pretrained_model=modelpath)
    return model

def train_StardistModel(train_imgs, train_masks, train_seg_args, model):
    
    path_save = train_seg_args['model_save_path']
    model_name = train_seg_args['model_name']
    
    model.basedir=path_save
    model.name=model_name
    model.train(train_imgs, train_masks, validation_data=(train_imgs,train_masks), epochs=2, steps_per_epoch=10)
    return model

from cv2 import GaussianBlur
def get_training_set(IMGS, Masks_stack, tz_actions, train_args):
    
    blur_args = train_args['blur']
    
    tzt = np.asarray(tz_actions)
    tzt = np.unique(tzt, axis=0)
    train_imgs = []
    train_masks = []
    
    for act in tzt:
        t,z = act
        img = IMGS[t,z]
        msk = Masks_stack[t,z]
        if blur_args is not None:
            img = GaussianBlur(img, blur_args[0], blur_args[1])
            msk = GaussianBlur(img, blur_args[0], blur_args[1])
        train_imgs.append(img)
        train_masks.append(msk)
    
    return train_imgs, train_masks

def check_train_segmentation_args(train_segmentation_args):
    if 'model_save_path' not in train_segmentation_args.keys():
        train_segmentation_args['model_save_path'] = None
        
    if 'model_name' not in train_segmentation_args.keys():
        train_segmentation_args['model_name'] = None
        
    if 'blur' not in train_segmentation_args.keys():
        train_segmentation_args['blur'] = None

from datetime import datetime
def fill_train_segmentation_args(train_segmentation_args, path_to_save, seg_args):
    
    if train_segmentation_args['model_save_path'] is None: train_segmentation_args['model_save_path'] = path_to_save
    if train_segmentation_args['model_name'] is None: 
        now = datetime.now()
        dt = now.strftime(seg_args['method']+"_%d-%m-%Y_%H-%M-%S")
        train_segmentation_args['model_name']=dt
    
    return train_segmentation_args