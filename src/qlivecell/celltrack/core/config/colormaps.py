# qlivecell/config/colormaps.py

import warnings
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Colormap


def _validate_rgb_triplets(x, name="masks_cmap"):
    if not isinstance(x, Iterable) or isinstance(x, (str, bytes)):
        raise TypeError(
            f"{name} must be a colormap name (str), a matplotlib Colormap, "
            "or an iterable of RGB triplets."
        )

    out = []
    for i, elem in enumerate(x):
        if not isinstance(elem, Iterable) or isinstance(elem, (str, bytes)):
            raise TypeError(f"{name}[{i}] must be an RGB triplet (iterable length 3).")
        try:
            r, g, b = elem
        except (TypeError, ValueError):
            raise ValueError(f"{name}[{i}] must have exactly 3 elements (r, g, b).")
        out.append((r, g, b))

    if len(out) == 0:
        raise ValueError(f"{name} custom colormap cannot be empty.")

    return out


def _as_rgb01_array(triplets, name="masks_cmap", warn=True) -> np.ndarray:
    arr = np.asarray(list(triplets), dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an iterable of RGB triplets (shape N x 3).")

    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))

    if min_val >= 0.0 and max_val <= 255.0 and max_val > 1.0:
        return arr / 255.0

    if min_val >= 0.0 and max_val <= 1.0:
        return arr

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
    continuous_samples=256,
    warn=True,
) -> ListedColormap:
    def _fallback():
        base = plt.get_cmap(fallback_name)
        cols = getattr(base, "colors", None)
        if cols is not None:
            return ListedColormap(np.asarray(cols), name=fallback_name)

        sampled = base(np.linspace(0, 1, continuous_samples, endpoint=True))[:, :3]
        return ListedColormap(sampled, name=fallback_name)

    # string name
    if isinstance(value, str):
        if value not in plt.colormaps():
            if warn:
                warnings.warn(
                    "invalid mask cmap, using tab10 instead; "
                    "run matplotlib.pyplot.colormaps() for the available colormaps",
                    UserWarning,
                )
            return _fallback()

        cmap = plt.get_cmap(value)
        cols = getattr(cmap, "colors", None)
        if cols is not None:
            return ListedColormap(np.asarray(cols), name=value)

        sampled = cmap(np.linspace(0, 1, continuous_samples, endpoint=True))[:, :3]
        return ListedColormap(sampled, name=value)

    # Colormap object
    if isinstance(value, Colormap):
        name = getattr(value, "name", "custom")
        cols = getattr(value, "colors", None)
        if cols is not None:
            return ListedColormap(np.asarray(cols), name=name)

        sampled = value(np.linspace(0, 1, continuous_samples, endpoint=True))[:, :3]
        return ListedColormap(sampled, name=name)

    # Custom triplets
    try:
        triplets = _validate_rgb_triplets(value)
        arr01 = _as_rgb01_array(triplets, warn=warn)
        return ListedColormap(arr01, name="custom_masks_cmap")
    except Exception:
        if warn:
            warnings.warn(
                "invalid mask cmap, using tab10 instead; masks_cmap must be a valid "
                "matplotlib colormap name, a Colormap, or an iterable of RGB triplets",
                UserWarning,
            )
        return _fallback()
