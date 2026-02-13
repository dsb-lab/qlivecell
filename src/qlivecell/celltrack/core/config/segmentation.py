# qlivecell/config/segmentation.py

import warnings

from .normalize import normalize_config
from .validators import v_one_of, v_int, v_bool_float_pair


def normalize_segmentation_and_args(
    segmentation_config,
    *,
    default_config,
    available_methods,
    get_default_args,  # inject dependency
):
    validators = {
        "method": v_one_of(available_methods, default=None),
        "min_outline_length": v_int(1, default=default_config["min_outline_length"]),
        "compute_center_method": v_one_of(
            ("centroid", "weighted_centroid"),
            default=default_config["compute_center_method"],
        ),
        "make_isotropic": v_bool_float_pair(default=tuple(default_config["make_isotropic"])),
    }

    def postprocess(cfg):
        # If method requires model but missing -> disable
        if cfg["method"] is not None and cfg.get("model", None) is None:
            warnings.warn(
                f"segmentation method {cfg['method']!r} requires a 'model'. Disabling segmentation.",
                UserWarning,
            )
            cfg["method"] = None

        # If 2D, force isotropic off
        if cfg["method"] is not None and "3D" not in cfg["method"]:
            cfg["make_isotropic"][0] = False

        if cfg["method"] is None:
            cfg["model"] = None

        return cfg

    seg_cfg = normalize_config(
        segmentation_config,
        default_config,
        validators=validators,
        postprocess=postprocess,
        name="segmentation_config",
        warn_unknown_keys=True,
    )

    # Build method args
    method = seg_cfg["method"]
    model = seg_cfg["model"]
    seg_method_args = {}

    if method is not None and model is not None:
        try:
            if "cellpose" in method:
                seg_method_args = get_default_args(model.eval)
            elif "stardist" in method:
                seg_method_args = get_default_args(model.predict_instances)
        except Exception as e:
            warnings.warn(
                f"could not extract default args from model for {method!r}: {e}. Using empty args.",
                UserWarning,
            )
            seg_method_args = {}

    # Allow user override of method args
    if isinstance(segmentation_config, dict):
        for k, v in segmentation_config.items():
            if k in seg_cfg:
                continue
            if k in seg_method_args:
                seg_method_args[k] = v
            else:
                warnings.warn(
                    f"unknown segmentation argument {k!r} for method {method!r}; ignoring.",
                    UserWarning,
                )

    return seg_cfg, seg_method_args
