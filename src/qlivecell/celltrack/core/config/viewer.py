# qlivecell/config/viewer.py

from .normalize import normalize_config
from .validators import v_tuple2_ints, v_int, v_one_of, v_bool_pair
from .colormaps import normalize_masks_cmap_to_listed


def normalize_viewer_config(
    viewer_config,
    *,
    default_config,
    CyclicList,  # pass it in (avoids circular imports)
):
    """
    Normalize viewer config + compute derived fields:
      - _cmap (ListedColormap)
      - labels_colors (CyclicList)
    """

    validators = {
        "layout": v_tuple2_ints(min_value=1, default=default_config["layout"]),
        "overlap": v_int(min_value=0, default=default_config["overlap"]),
        "display_scaling": v_int(min_value=0, default=default_config["display_scaling"]),
        "display_centers": v_bool_pair(default=tuple(default_config["display_centers"])),
        "backup_steps": v_int(min_value=0, default=default_config["backup_steps"]),
        "wheel_motor": v_one_of(("regular", "fancy"), default=default_config["wheel_motor"]),
        "line_builder_mode": v_one_of(("lasso", "line"), default=default_config["line_builder_mode"]),
        "min_outline_length": v_int(min_value=1, default=default_config["min_outline_length"]),
        # channels: you may validate separately depending on your stack/channels logic
    }

    def postprocess(cfg):
        cmap = normalize_masks_cmap_to_listed(cfg["masks_cmap"])
        cfg["_cmap"] = cmap
        cfg["labels_colors"] = CyclicList(cmap.colors)
        return cfg

    return normalize_config(
        viewer_config,
        default_config,
        validators=validators,
        postprocess=postprocess,
        name="viewer_config",
        warn_unknown_keys=True,
    )
