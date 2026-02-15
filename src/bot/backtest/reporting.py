from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v) for v in value]

    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if value is None:
        return None

    if isinstance(value, (pd.Timestamp,)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    if isinstance(value, (pd.Series, pd.Index)):
        return [sanitize_for_json(v) for v in value.tolist()]

    if isinstance(value, np.ndarray):
        return [sanitize_for_json(v) for v in value.tolist()]

    return value


def dumps_strict_json(payload: Any, *, indent: int = 2) -> str:
    cleaned = sanitize_for_json(payload)
    return json.dumps(cleaned, indent=indent, allow_nan=False)


def write_strict_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_strict_json(payload, indent=indent), encoding="utf-8")
    return out
