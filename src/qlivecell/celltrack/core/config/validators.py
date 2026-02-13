# qlivecell/config/validators.py

import warnings
from .normalize import ValidatorResult


def v_one_of(options, *, default=None, warn=True):
    options = tuple(options)

    def _v(value, cfg, key):
        if value in (None, "", "none", "None"):
            return ValidatorResult(default if default is not None else None)

        if value not in options:
            if warn:
                warnings.warn(
                    f"{key} must be one of {list(options)}; got {value!r}. Using default.",
                    UserWarning,
                )
            return ValidatorResult(default)

        return ValidatorResult(value)

    return _v


def v_int(min_value=None, *, default=None, warn=True):
    def _v(value, cfg, key):
        try:
            iv = int(value)
        except Exception:
            if warn:
                warnings.warn(f"{key} must be an int; got {value!r}. Using default.", UserWarning)
            return ValidatorResult(default)

        if min_value is not None and iv < min_value:
            if warn:
                warnings.warn(
                    f"{key} must be >= {min_value}; got {iv}. Using default.",
                    UserWarning,
                )
            return ValidatorResult(default)

        return ValidatorResult(iv)

    return _v


def v_tuple2_ints(min_value=1, *, default=(1, 1), warn=True):
    def _v(value, cfg, key):
        try:
            a, b = value
            a = int(a)
            b = int(b)
        except Exception:
            if warn:
                warnings.warn(
                    f"{key} must be a tuple/list of 2 ints. Got {value!r}. Using default.",
                    UserWarning,
                )
            return ValidatorResult(tuple(default))

        if a < min_value or b < min_value:
            if warn:
                warnings.warn(
                    f"{key} entries must be >= {min_value}. Got {value!r}. Using default.",
                    UserWarning,
                )
            return ValidatorResult(tuple(default))

        return ValidatorResult((a, b))

    return _v


def v_bool_pair(*, default=(True, True), warn=True):
    def _v(value, cfg, key):
        try:
            a, b = value
        except (TypeError, ValueError):
            if warn:
                warnings.warn(
                    f"{key} must be an iterable of length 2: [bool, bool]. Got {value!r}. Using default.",
                    UserWarning,
                )
            return ValidatorResult([bool(default[0]), bool(default[1])])

        if not isinstance(a, bool) or not isinstance(b, bool):
            if warn:
                warnings.warn(
                    f"{key} must be [bool, bool]. Got {value!r}. Using default.",
                    UserWarning,
                )
            return ValidatorResult([bool(default[0]), bool(default[1])])

        return ValidatorResult([a, b])

    return _v


def v_bool_float_pair(*, default=(False, 1.0), warn=True):
    def _v(value, cfg, key):
        try:
            a, b = value
        except (TypeError, ValueError):
            if warn:
                warnings.warn(
                    f"{key} must be an iterable of length 2: [bool, float]. Got {value!r}. Using default.",
                    UserWarning,
                )
            return ValidatorResult([bool(default[0]), float(default[1])])

        if not isinstance(a, bool) or not isinstance(b, (int, float)):
            if warn:
                warnings.warn(
                    f"{key} must be [bool, float]. Got {value!r}. Using default.",
                    UserWarning,
                )
            return ValidatorResult([bool(default[0]), float(default[1])])

        return ValidatorResult([bool(a), float(b)])

    return _v
