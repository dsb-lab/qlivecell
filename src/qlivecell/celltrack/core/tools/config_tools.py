import matplotlib
from ..plot.plot_iters import CyclicList
from .ct_tools import get_cell_color, set_cell_color
import numpy as np
from collections.abc import Iterable

from qlivecell.config import DEFAULT_VIEWER_CONFIG
import warnings
import difflib

def merge_config(default_config, user_config, *, config_name="config", strict=False):
    """
    Merge user_config into default_config.

    Parameters
    ----------
    default_config : dict
        Dictionary containing default settings.
    user_config : dict
        User-provided settings.
    config_name : str
        Name used in warning/error messages.
    strict : bool
        If True, raise ValueError on unknown keys instead of warning.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """

    merged = default_config.copy()

    for key, value in user_config.items():

        if key not in default_config:
            suggestion = difflib.get_close_matches(key, default_config.keys(), n=1)

            if suggestion:
                message = (
                    f"'{key}' is not a valid {config_name} option. "
                    f"Did you mean '{suggestion[0]}'?"
                )
            else:
                message = f"'{key}' is not a valid {config_name} option."

            if strict:
                raise ValueError(message)
            else:
                warnings.warn(message, UserWarning)
                continue

        merged[key] = value

    return merged


def validate_and_fill_viewer_config(user_viewer_config, stack_dims, channels_order):

    viewer_config = merge_config(DEFAULT_VIEWER_CONFIG, user_viewer_config, config_name="viewer config")
    
    # Make sure settings make sense, use default otherwise and raise a warning
    
    # plot layout must be iterable. I should change this to 
    if not hasattr(viewer_config["layout"], "__iter__"):
        warnings.warn("invalid layout, using (1,1) instead", UserWarning)
        viewer_config["layout"] = (1, 1)

    if np.multiply(*viewer_config["layout"]) >= viewer_config["overlap"]:
        viewer_config["overlap"] = np.multiply(*viewer_config["layout"]) - 1
        
    if viewer_config["masks_cmap"] not in matplotlib.pyplot.colormaps():
        warnings.warn("invalid mask cmap, using tab10 instead, run matplotlib.pyplot.colormaps() for the available colormaps", UserWarning)
        viewer_config["masks_cmap"] = "tab10"
    
    if not is_bool_pair(viewer_config["display_centers"]):
        warnings.warn("invalid display_centers setting, using [True, True] instead.", UserWarning)
    
    if viewer_config["channels"] is None:
        viewer_config["channels"] = channels_order[:1]

    if viewer_config["display_scaling"] is None:
        viewer_config["display_scaling"] = 1
    
    viewer_config["plot_stack_dims"] = np.rint(np.array(stack_dims)*viewer_config["display_scaling"]).astype("int32")
    
    _cmap = normalize_masks_cmap_to_listed(viewer_config["masks_cmap"])
    viewer_config["labels_colors"] = CyclicList(_cmap.colors)
    viewer_config["plot_masks"] = True

    # Interaction settings
    if viewer_config["wheel_motor"] not in ["regular", "smooth"]:
        warnings.warn("invalid wheel_motor setting, using regular instead. Valid options are [regular, smooth]", UserWarning)
        viewer_config["wheel_motor"] = "regular"
    
    if viewer_config["line_builder_mode"] not in ["points", "lasso"]:
        warnings.warn("invalid line_builder_mode setting, using points instead. Valid options are [points, lasso]", UserWarning)
        viewer_config["line_builder_mode"] = "points"
    
    return viewer_config

def is_bool_pair(x):
    try:
        a, b = x
        return isinstance(a, bool) and isinstance(b, bool)
    except (TypeError, ValueError):
        return False

import warnings
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Colormap


def _validate_rgb_triplets(x, name="masks_cmap"):
    """
    Validate that x is an iterable of iterables of length 3.
    Returns a list of (r,g,b) triplets (values not normalized yet).
    Accepts lists/tuples/np arrays/generators.
    Excludes strings/bytes.
    """
    if not isinstance(x, Iterable) or isinstance(x, (str, bytes)):
        raise TypeError(
            f"{name} must be a colormap name (str), a matplotlib Colormap, "
            "or an iterable of RGB triplets."
        )

    out = []
    for i, elem in enumerate(x):
        if not isinstance(elem, Iterable) or isinstance(elem, (str, bytes)):
            raise TypeError(f"{name}[{i}] must be an RGB triplet (iterable of length 3).")
        try:
            r, g, b = elem
        except (TypeError, ValueError):
            raise ValueError(f"{name}[{i}] must have exactly 3 elements (r, g, b).")
        out.append((r, g, b))

    if len(out) == 0:
        raise ValueError(f"{name} custom colormap cannot be empty.")

    return out


def _as_rgb01_array(triplets, name="masks_cmap", warn=True) -> np.ndarray:
    """
    Convert RGB triplets to a float array in [0,1].
    Accepts either [0,1] floats or [0,255] ints/floats.
    If values are outside both valid ranges, emits a warning and raises.
    """
    arr = np.asarray(list(triplets), dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an iterable of RGB triplets (shape N x 3).")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} custom colormap cannot be empty.")

    max_val = float(np.nanmax(arr))
    min_val = float(np.nanmin(arr))

    # Case 1: looks like 0..255 range (non-negative, max <= 255, and > 1 somewhere)
    if min_val >= 0.0 and max_val <= 255.0 and max_val > 1.0:
        arr = arr / 255.0
        return arr

    # Case 2: valid 0..1 range
    if min_val >= 0.0 and max_val <= 1.0:
        return arr

    # Case 3: invalid -> warn + raise (caller will fallback)
    if warn:
        warnings.warn(
            f"{name} RGB values must be in [0,1] or [0,255]. "
            f"Got range [{min_val:.3f}, {max_val:.3f}]. Using fallback colormap instead.",
            UserWarning,
        )
    raise ValueError("Invalid RGB range")


def normalize_masks_cmap_to_listed(
    value,
    *,
    fallback_name="tab10",
    continuous_samples=20,  # only used for continuous colormaps (e.g., viridis)
    warn=True,
) -> ListedColormap:
    """
    Normalize masks_cmap input into a ListedColormap so downstream can do `_cmap.colors`.

    Accepted inputs:
      1) str: a matplotlib colormap name
         - if discrete (has .colors): uses original colors exactly
         - if continuous: samples `continuous_samples` colors in [0,1]
      2) matplotlib.colors.Colormap: same as above
      3) iterable of RGB triplets (N x 3): values in [0,1] or [0,255]
         - if invalid RGB range: warns and falls back
         - returned cmap uses exactly those colors (length N)

    On any invalid input: warns (if warn=True) and returns fallback cmap as ListedColormap.
    """

    def _fallback():
        base = plt.get_cmap(fallback_name)
        cols = getattr(base, "colors", None)
        if cols is None:
            cols = base(np.linspace(0, 1, continuous_samples, endpoint=True))[:, :3]
        return ListedColormap(np.asarray(cols), name=fallback_name)

    # 1) string name
    if isinstance(value, str):
        if value not in plt.colormaps():
            if warn:
                warnings.warn(
                    "invalid mask cmap, using tab10 instead; run matplotlib.pyplot.colormaps() "
                    "for the available colormaps",
                    UserWarning,
                )
            return _fallback()

        cmap = plt.get_cmap(value)
        cols = getattr(cmap, "colors", None)

        # Discrete cmap: preserve exact palette and length
        if cols is not None:
            return ListedColormap(np.asarray(cols), name=value)

        # Continuous cmap: sample
        cols = cmap(np.linspace(0, 1, continuous_samples, endpoint=True))[:, :3]
        return ListedColormap(np.asarray(cols), name=value)

    # 2) matplotlib Colormap object
    if isinstance(value, Colormap):
        cols = getattr(value, "colors", None)
        name = getattr(value, "name", "custom")

        if cols is not None:
            return ListedColormap(np.asarray(cols), name=name)

        cols = value(np.linspace(0, 1, continuous_samples, endpoint=True))[:, :3]
        return ListedColormap(np.asarray(cols), name=name)

    # 3) custom RGB triplets
    try:
        triplets = _validate_rgb_triplets(value, name="masks_cmap")
        arr01 = _as_rgb01_array(triplets, name="masks_cmap", warn=warn)
        return ListedColormap(arr01, name="custom_masks_cmap")
    except Exception:
        if warn:
            warnings.warn(
                "invalid mask cmap, using tab10 instead; masks_cmap must be a valid "
                "matplotlib colormap name, a Colormap, or an iterable of RGB triplets "
                "(values in [0,1] or [0,255])",
                UserWarning,
            )
        return _fallback()
