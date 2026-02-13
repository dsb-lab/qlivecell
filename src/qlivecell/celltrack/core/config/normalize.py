# qlivecell/config/normalize.py

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ValidatorResult:
    value: Any
    ok: bool = True


Validator = Callable[[Any, dict, str], ValidatorResult]


def normalize_config(
    user_config: Any,
    default_config: dict,
    *,
    validators: dict[str, Validator] | None = None,
    postprocess: Callable[[dict], dict] | None = None,
    name: str = "config",
    warn_unknown_keys: bool = True,
) -> dict:
    """
    Generic check+fill+warn pipeline.

    Steps:
      1) deep-copy defaults
      2) merge user overrides (unknown keys -> warn+ignore)
      3) run per-key validators (validator errors -> warn + reset to default)
      4) run postprocess for cross-field logic
      5) return canonical config
    """
    cfg = copy.deepcopy(default_config)
    validators = validators or {}

    # 1) sanitize user_config
    if user_config is None:
        user_config = {}
    if not isinstance(user_config, dict):
        warnings.warn(
            f"{name} must be a dict; got {type(user_config).__name__}. Using defaults.",
            UserWarning,
        )
        user_config = {}

    # 2) merge overrides
    for k, v in user_config.items():
        if k in cfg:
            cfg[k] = v
        else:
            if warn_unknown_keys:
                warnings.warn(f"unknown {name} key {k!r}; ignoring.", UserWarning)

    # 3) validate/coerce
    for k, fn in validators.items():
        if k not in cfg:
            continue
        try:
            res = fn(cfg[k], cfg, k)
            cfg[k] = res.value
        except Exception as e:
            warnings.warn(f"error validating {name}.{k}: {e}. Using default.", UserWarning)
            cfg[k] = copy.deepcopy(default_config[k])

    # 4) postprocess (cross-field)
    if postprocess is not None:
        try:
            cfg = postprocess(cfg)
        except Exception as e:
            warnings.warn(f"error in {name} postprocess: {e}.", UserWarning)

    return cfg
