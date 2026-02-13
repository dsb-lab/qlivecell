# qlivecell

**qlivecell** is an open-source Python package for **semi-automatic cell segmentation and tracking in multichannel 2D/3D time-lapse microscopy**, with a strong focus on **fast manual curation** and **annotation of cellular events** (e.g., mitosis and apoptosis).

It provides a modular pipeline that combines state-of-the-art segmentation methods with point-based tracking, efficient handling of large movies through batch processing, and an interactive user interface designed to quickly correct segmentation/tracking errors.

> Repository: https://github.com/dsb-lab/qlivecell

---

## Key features

- **2D and 3D segmentation workflows**
  - Integrates **StarDist** and **Cellpose** segmentation models.
  - Supports a **2D-to-3D concatenation** strategy for anisotropic stacks (segment in 2D, link masks across z).

- **3D mask building + automatic correction utilities**
  - Slice connection across z using a distance-based linking criterion.
  - Detection and correction of over-concatenated masks using combined intensity/overlap profiles.
  - Tools to remove **debris** (size filtering), and to split long/overlapping masks using length-based thresholds.

- **Cell tracking**
  - **Greedy tracking** (fast, distance-based).
  - **Hungarian tracking** (distance + volume + shape; configurable weights).

- **Interactive curation & annotation UI**
  - Matplotlib-based GUI with keyboard-driven actions for efficient curation.
  - Visualization of masks/outlines/centers over raw data and multichannel inspection.
  - Support for annotation of events in time and space.

- **Batch mode for large datasets**
  - Processes data in time batches to reduce memory usage.
  - Propagates label corrections forward in time via correspondence structures.

- **Interoperability**
  - Outputs stored in portable formats (e.g., NumPy `.npy` label masks), easy to use with external tools.
  - Optional integration with **napari** for visualization workflows.

---

## Installation

We recommend using a fresh virtual environment (e.g., `pyenv`, `venv`, or `conda`).

### Install from GitHub
```bash
pip install git+https://github.com/dsb-lab/qlivecell.git
```

---

## Quickstart

The minimal workflow with **qlivecell** consists of:

1. Initializing a `cellSegTrack` object
2. Running or loading segmentation & tracking
3. Inspecting and curating results using the built-in GUI

```python
from qlivecell import cellSegTrack

cST = cellSegTrack(
    path_data="path/to/image_data",
    path_save="path/to/output",
    channels=[0], 
    segmentation_config={...},
    concatenation3D_args={...},
    tracking_args={...},
    batch_args={...},
    viewer_config={...}
)

# Run segmentation + tracking
cST.run()

# or load existing results
# cST.load()

# Launch interactive curation 
# cST.plot_tracking()

```

## Examples

For now, the `examples/` folder contains a single minimal, end-to-end example:

- `examples/segmentation_tracking.py`

This script is the recommended starting point. It demonstrates:
- how to initialize `cellSegTrack`,
- how to configure `segmentation_config` for **StarDist** and **Cellpose**,
- how to run segmentation and tracking,
- and how to launch the interactive curation GUI.

More examples will be provided shortly.

---

## Documentation

A full, stable API reference is currently under active development.

In the meantime, users are encouraged to refer to the `examples/` folder, which contains
fully working and well-annotated scripts illustrating all major workflows
(segmentation, tracking, curation, and embryo segmentation).
These examples should be considered the primary usage guide for the current release.
