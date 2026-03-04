from __future__ import annotations

import os
from typing import Any, Dict
from pathlib import Path
import yaml


def project_root() -> str:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return env_root
    return str(Path(__file__).resolve().parent.parent)


def resolve_path(path: str) -> str:
    if path is None:
        raise ValueError("path is None")

    expanded = os.path.expanduser(path)
    p = Path(expanded)
    if p.is_absolute():
        return str(p)

    return str(Path(project_root()) / p)


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str) -> None:
    os.makedirs(resolve_path(p), exist_ok=True)
