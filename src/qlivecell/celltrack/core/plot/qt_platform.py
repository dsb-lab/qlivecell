import os
import sys
from contextlib import contextmanager


@contextmanager
def qt_platform():
    """Prefer native Wayland while Qt initializes on a Linux Wayland session."""
    if (
        not sys.platform.startswith("linux")
        or "QT_QPA_PLATFORM" in os.environ
        or not (
            os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE") == "wayland"
        )
    ):
        yield
        return

    os.environ["QT_QPA_PLATFORM"] = (
        "wayland;xcb" if os.environ.get("DISPLAY") else "wayland"
    )
    try:
        yield
    finally:
        os.environ.pop("QT_QPA_PLATFORM", None)
