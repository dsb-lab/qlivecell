import shutil

import numpy as np
import skimage

from ..tools.tools import get_default_args


def train_CellposeModel(train_imgs, train_masks, model, train_seg_args):
    from cellpose import models

    new_model = models.CellposeModel(
        gpu=model.gpu, pretrained_model=model.pretrained_model[0]
    )

    modelpath = new_model.train(train_imgs, train_masks, **train_seg_args)
    new_model = models.CellposeModel(gpu=model.gpu, pretrained_model=modelpath)
    return new_model


def train_StardistModel(train_imgs, train_masks, model, train_seg_args):
    config = train_seg_args.pop("config")
    name = train_seg_args.pop("name")
    basedir = train_seg_args.pop("basedir")
    train_new_model = train_seg_args.pop("train_new_model")

    if train_new_model:
        new_model = model.__class__(config, name=name, basedir=basedir)
    else:
        shutil.copytree(model.logdir, basedir + name, dirs_exist_ok=True)
        new_model = model.__class__(None, name=name, basedir=basedir)
        new_model.config = config
        new_model._update_and_check_config()

    new_model.train(
        train_imgs,
        train_masks,
        validation_data=(train_imgs, train_masks),
        **train_seg_args,
    )
    return new_model


def blur_imgs(img, blur_args):
    blur_img = img.copy()
    if len(img.shape) == 3:
        for ch in range(img.shape[-1]):
            blur_img[:, :, ch] = skimage.filters.gaussian(
                img[:, :, ch], sigma=blur_args[0], truncate=blur_args[1]
            )
    else:
        blur_img[:, :, ch] = skimage.filters.gaussian(
            img, sigma=blur_args[0], truncate=blur_args[1]
        )
    return blur_img


def get_training_set(IMGS, Masks_stack, tz_actions, train_args, train3D=False):
    blur_args = train_args["blur"]

    tzt = np.asarray(tz_actions)
    tzt = np.unique(tzt, axis=0)
    train_imgs = []
    train_masks = []

    for act in tzt:
        t, z = act

        if train3D:
            img = IMGS[t]
            msk = Masks_stack[t]
        else:
            img = IMGS[t, z]
            msk = Masks_stack[t, z]
            if blur_args is not None:
                img = blur_imgs(img, blur_args)
        train_imgs.append(img)
        train_masks.append(msk)

    return train_imgs, train_masks


from datetime import datetime


def check_and_fill_train_segmentation_args(
    train_segmentation_args, model, seg_method, path_to_save
):
    if model is None:
        return None, None
    else:
        train_seg_args = get_default_args(model.train)

    new_train_segmentation_args = {
        "blur": None,
    }

    # Define model save dir and name for all segmentation methods
    if "cellpose" in seg_method:
        path_save_arg = "save_path"
        model_name_arg = "model_name"

        if path_save_arg not in train_segmentation_args.keys():
            train_segmentation_args[path_save_arg] = path_to_save

        if (
            model_name_arg not in train_segmentation_args.keys()
            or train_segmentation_args[model_name_arg] is None
        ):
            now = datetime.now()
            dt = now.strftime(seg_method + "_%d-%m-%Y_%H-%M-%S")
            train_segmentation_args[model_name_arg] = dt

        config_args_dict = {}

    elif "stardist" in seg_method:
        path_save_arg = "basedir"
        model_name_arg = "name"

        if path_save_arg not in train_segmentation_args.keys():
            train_segmentation_args[path_save_arg] = path_to_save

        if (
            model_name_arg not in train_segmentation_args.keys()
            or train_segmentation_args[model_name_arg] is None
        ):
            now = datetime.now()
            dt = now.strftime(seg_method + "_%d-%m-%Y_%H-%M-%S")
            train_segmentation_args[model_name_arg] = dt

        if "train_new_model" not in train_segmentation_args.keys():
            train_segmentation_args["train_new_model"] = False

        train_seg_args["basedir"] = train_segmentation_args.pop(path_save_arg)
        train_seg_args["name"] = train_segmentation_args.pop(model_name_arg)
        config_args_dict = model.config.__dict__
        train_seg_args["config"] = model.config
        train_seg_args["train_new_model"] = train_segmentation_args.pop(
            "train_new_model"
        )

    for tsarg in train_segmentation_args.keys():
        if tsarg in new_train_segmentation_args.keys():
            new_train_segmentation_args[tsarg] = train_segmentation_args[tsarg]
        elif tsarg in train_seg_args.keys():
            train_seg_args[tsarg] = train_segmentation_args[tsarg]
        # In the case of Stardist, most training arguments are part of the model config
        elif tsarg in config_args_dict.keys():
            train_seg_args["config"].__dict__[tsarg] = train_segmentation_args[tsarg]
        else:
            raise Exception(
                "key %s is not a correct training argument for the selected segmentation method"
                % tsarg
            )

    return new_train_segmentation_args, train_seg_args
