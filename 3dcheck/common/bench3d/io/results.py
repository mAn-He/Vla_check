"""summary.json: one file per (notebook, model[, variant]) with a fixed schema.

Schema (version 1)
------------------
{
  "schema": 1,
  "part": "outdoor" | "indoor",
  "notebook": "02_monocular_depth",
  "model": "da3_metric_large",           # short id used in tables
  "variant": "cam_front/eval4",           # free text: input setting
  "metrics": {"abs_rel": 0.12, ...},      # flat name -> float (nested dicts go to "extra")
  "runtime_s": 12.3, "peak_vram_mb": 4567.0,
  "checkpoint": "depth-anything/DA3METRIC-LARGE",
  "packages": {"torch": "2.1.2", ...},
  "seed": 0, "profile": "mid",
  "extra": {...}, "created": "2026-10-17T12:00:00"
}
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

SUMMARY_SCHEMA_VERSION = 1
_REQUIRED = ("model", "metrics", "runtime_s", "peak_vram_mb", "checkpoint", "packages", "seed")


def result_dir(cfg, notebook: str, model: str | None = None) -> Path:
    """results/<part>/<notebook>/<model>/ (created)."""
    d = cfg.path(cfg["results_root"]) / notebook
    if model:
        d = d / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_cached(path: str | Path, force_rerun: bool = False) -> bool:
    """True if `path` exists and FORCE_RERUN is off (notebooks skip recomputation)."""
    return (not force_rerun) and Path(path).exists()


def _clean(v):
    """Make numpy scalars / nan JSON friendly."""
    try:
        import numpy as np

        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:
        pass
    if isinstance(v, float) and v != v:  # nan
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def save_summary(
    out_dir: str | Path,
    *,
    part: str,
    notebook: str,
    profile: str,
    variant: str = "",
    extra: dict | None = None,
    filename: str = "summary.json",
    **fields,
) -> Path:
    """Write summary.json.

    Required keyword fields: model, metrics, runtime_s, peak_vram_mb, checkpoint, packages, seed.
    """
    missing = [k for k in _REQUIRED if k not in fields]
    if missing:
        raise ValueError(f"summary missing fields: {missing}")
    metrics = fields["metrics"]
    bad = {k: v for k, v in metrics.items() if isinstance(v, (dict, list))}
    if bad:
        raise ValueError(f"metrics must be flat name->number; move these to extra: {list(bad)}")
    rec = {
        "schema": SUMMARY_SCHEMA_VERSION,
        "part": part,
        "notebook": notebook,
        "variant": variant,
        "profile": profile,
        **fields,
        "extra": extra or {},
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(out_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(rec), indent=2, ensure_ascii=False))
    return out


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def collect_summaries(results_root: str | Path, pattern: str = "**/summary*.json"):
    """Long-format DataFrame: one row per (summary file, metric)."""
    import pandas as pd

    rows = []
    for f in sorted(Path(results_root).glob(pattern)):
        s = load_summary(f)
        base = {
            k: s.get(k)
            for k in ("part", "notebook", "model", "variant", "profile", "runtime_s", "peak_vram_mb", "checkpoint")
        }
        base["file"] = str(f)
        for name, val in s.get("metrics", {}).items():
            rows.append({**base, "metric": name, "value": val})
    return pd.DataFrame(rows)
