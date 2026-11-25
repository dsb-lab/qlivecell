# Global configuration for qlivecell

from importlib.metadata import PackageNotFoundError, version

PROGRESS = True
CLEARPRINTS = True


def _check_tensorflow_distributions():
    try:
        version("tensorflow")
        version("tensorflow-cuda13")
    except PackageNotFoundError:
        return

    raise ImportError(
        "tensorflow and tensorflow-cuda13 both install the tensorflow module. "
        "Create a fresh environment and select one qlivecell TensorFlow extra: "
        "tensorflow, cuda12, or cuda13."
    )
