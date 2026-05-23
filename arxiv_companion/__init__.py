"""arXiv Research Companion — hybrid vector + citation-graph RAG.

Top-level package. Re-exports the main entry points so callers can do:

    from arxiv_companion import build_index, query, evaluate
"""
from pathlib import Path
from typing import Any

import yaml

__version__ = "0.1.0"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config. Falls back to the bundled config.yaml at repo root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["load_config", "DEFAULT_CONFIG_PATH", "__version__"]
