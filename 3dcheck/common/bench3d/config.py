"""Config loading and repo-root resolution shared by all notebooks."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


def find_root(start: str | Path | None = None) -> Path:
    """Return the 3dcheck root (the folder holding envs/ and common/bench3d/).

    Honors $THREEDCHECK_ROOT, otherwise walks up from `start` (default: cwd).
    """
    env = os.environ.get("THREEDCHECK_ROOT")
    if env:
        return Path(env).resolve()
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "envs").is_dir() and (cand / "common" / "bench3d").is_dir():
            return cand
    raise FileNotFoundError("3dcheck root not found; set THREEDCHECK_ROOT or run from inside the repo")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict with helpers. `cfg.p` is the active hardware profile, `cfg.path(rel)` resolves repo-relative paths."""

    @property
    def root(self) -> Path:
        return Path(self["_root"])

    @property
    def p(self) -> dict:
        return self["_profile_values"]

    def path(self, rel: str | Path) -> Path:
        rel = Path(rel)
        return rel if rel.is_absolute() else (self.root / rel)

    def third_party(self, name: str = "") -> Path:
        """Pinned upstream repos (envs/third_party.lock); $THIRD_PARTY overrides the config value."""
        base = (
            Path(os.environ["THIRD_PARTY"])
            if os.environ.get("THIRD_PARTY")
            else self.path(self.get("third_party", "third_party"))
        )
        return base / name if name else base


def load_config(part: str = "outdoor", profile: str | None = None, path: str | Path | None = None) -> Config:
    """Load `<root>/<part>/config.yaml` and resolve the active profile.

    Profile priority: argument > $THREEDCHECK_PROFILE > `profile:` in the yaml.
    The `profiles.<name>` block is deep-merged over `profiles.mid` so a profile only lists what differs.
    """
    root = find_root()
    path = Path(path) if path else root / part / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    prof = profile or os.environ.get("THREEDCHECK_PROFILE") or raw.get("profile", "mid")
    profiles = raw.get("profiles", {})
    if prof not in profiles:
        raise KeyError(f"profile '{prof}' not in {list(profiles)}")
    values = profiles[prof] if prof == "mid" else _deep_merge(profiles.get("mid", {}), profiles[prof])
    cfg = Config(raw)
    cfg["_root"] = str(root)
    cfg["_part"] = part
    cfg["profile"] = prof
    cfg["_profile_values"] = values
    return cfg
