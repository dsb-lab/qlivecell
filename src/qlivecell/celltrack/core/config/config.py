# Global configuration for qlivecell

PROGRESS = True
CLEARPRINTS = True

DEFAULT_VIEWER_CONFIG = {
    # display
    "layout": (1, 1),
    "overlap": 0,
    "masks_cmap": "tab10",
    "display_scaling": 1,
    "display_centers": [True, True],
    "channels": None,
    "min_outline_length": 1,
    # interaction
    "wheel_motor": "regular",
    "backup_steps": 10,
    "line_builder_mode": "lasso",
}

DEFAULT_SEGMENTATION_CONFIG = {
    "method": None,
    "model": None,
    "blur": None,
    "make_isotropic": [False, 1.0],
    "min_outline_length": 1,
    "compute_center_method": "weighted_centroid",
}

AVAILABLE_SEGMENTATION_METHODS = (
    "cellpose2D",
    "cellpose3D",
    "stardist2D",
    "stardist3D",
)