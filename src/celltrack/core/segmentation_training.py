import numpy as np

def train_CellposeModel(train_imgs, train_masks, model, channels, path_save, model_name):    
    from cellpose import models
    model  = models.CellposeModel(gpu=model.gpu, pretrained_model=model.pretrained_model[0])
    modelpath = model.train(train_imgs, train_masks, channels = channels, save_path=path_save, model_name=model_name)
    model  = models.CellposeModel(gpu=model.gpu, pretrained_model=modelpath)
    return model

def train_StardistModel(IMGS, Masks_stack, tz_actions, model, path_save, model_name):
    pass

def get_training_set(IMGS, Masks_stack, tz_actions):
    tzt = np.asarray(tz_actions)
    tzt = np.unique(tzt, axis=0)
    train_imgs = []
    train_masks = []
    
    for act in tzt:
        t,z = act
        train_imgs.append(IMGS[t,z])
        train_masks.append(Masks_stack[t,z])
    
    return train_imgs, train_masks
