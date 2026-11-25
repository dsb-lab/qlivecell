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

qlivecell 0.1.0 requires Python 3.11 or later. From a checkout, create and
activate a virtual environment, then install the features you use.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The base install supports tracking, saved results, and Matplotlib. Add `qt`
for PyQt6, `napari` for the 3D viewer, or `cellpose` for Cellpose. For
StarDist, select its extra and **one** TensorFlow extra per environment.

| StarDist environment | Install from the checkout |
| --- | --- |
| CPU | `python -m pip install -e '.[stardist,tensorflow,qt]'` |
| CUDA 12 (Linux) | `python -m pip install -e '.[stardist,cuda12,qt]'` |
| CUDA 13 (Linux, release candidate or newer) | `python -m pip install -e '.[stardist,cuda13,qt]'` |

Use separate environments when switching between CUDA 13 and the standard
TensorFlow distribution. pip combines requested extras, and both distributions
write the same `tensorflow` package files. qlivecell detects the overlapping
installs at import. StarDist also needs a model checkpoint, which
`StarDist2D.from_pretrained(...)` downloads on first use. [TensorFlow installation guide](https://www.tensorflow.org/install/pip) · [CUDA 13 wheels](https://pypi.org/project/tensorflow-cuda13/)

Cellpose `2.x` and `3.x` use their legacy channel configuration. With
Cellpose `4.x`, use `models.CellposeModel`. The model receives channels in
input order by default. To select channels explicitly, pass a legacy channel
pair in `segmentation_args`. qlivecell applies that selection before
inference because Cellpose 4 ignores the old `channels` parameter. Cellpose 4
uses different checkpoints and
model architectures, so upgrading the model can change masks. Use
`gpu=False` for CPU inference. For GPU inference, install the PyTorch
build for your driver and platform before adding qlivecell's `cellpose`
extra, then verify `torch.cuda.is_available()`. The TensorFlow extras
configure StarDist's runtime. [Cellpose settings](https://cellpose.readthedocs.io/en/latest/settings.html) · [PyTorch installation](https://pytorch.org/get-started/locally/)

### Linux display troubleshooting

When qlivecell launches its Matplotlib curation or napari viewer on a Wayland
session, it prefers Qt's native Wayland plugin and includes XCB as a fallback
if `DISPLAY` is set. On X11, Qt uses its default platform. qlivecell honors
`QT_QPA_PLATFORM` and the Matplotlib backend already selected by the caller.

- If you already have a Qt application, set its platform before that application
  starts. To diagnose plugin loading, launch the script with
  `QT_DEBUG_PLUGINS=1 QT_QPA_PLATFORM=wayland python your_script.py`.
- On X11 or XWayland, try `QT_QPA_PLATFORM=xcb` for a single launch. Qt 6's
  XCB plugin needs the system X11 libraries, including `libxcb-cursor0` on
  Ubuntu. If a runtime library is missing, use native Wayland where available
  or ask the system administrator to provide the library.
- Use `MPLBACKEND=Agg` for headless Matplotlib runs. For Qt tests without a
  display, use `QT_QPA_PLATFORM=offscreen MPLBACKEND=QtAgg`.

[Qt platform selection](https://doc.qt.io/qt-6/qpa.html#selecting-a-qpa-plugin) · [Qt Linux runtime requirements](https://doc.qt.io/qt-6/linux-requirements.html)

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
    segmentation_args={...},
    concatenation3D_args={...},
    tracking_args={...},
    error_correction_args={...},
    batch_args={...},
    plot_args={...}
)

# Run segmentation + tracking
cST.run()

# or load existing results
# cST.load()

# Launch interactive curation 
# cST.plot()

```

## Examples

The scripts under [examples/artifitial_data](examples/artifitial_data) generate
synthetic 2D and 3D TIFF time series and run StarDist segmentation, tracking,
and manual curation. Run the generator for the desired dimensionality before
the matching tracking script.

---

## Documentation

A full, stable API reference is currently under active development.

In the meantime, users are encouraged to refer to the `examples/` folder, which contains
fully working and well-annotated scripts illustrating all major workflows
(segmentation, tracking, curation, and embryo segmentation).
These examples should be considered the primary usage guide for the current release.
