from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from .io import atomic_write_json


CANDIDATES = {
    "chronos_2": ["chronos", "chronos2"],
    "tirex_2": ["tirex", "tirex2"],
    "timesfm": ["timesfm"],
}


def local_pretrained_preflight(root: Path, output_name: str = "pretrained_local_check_v2.json") -> Path:
    cache = root / "artifacts/pretrained_cache"
    checkpoint_files = [str(path.relative_to(root)) for path in cache.rglob("*") if path.is_file()] if cache.exists() else []
    results: dict[str, Any] = {}
    for candidate, modules in CANDIDATES.items():
        installed = [module for module in modules if importlib.util.find_spec(module) is not None]
        results[candidate] = {
            "package_candidates": modules,
            "installed_candidates": installed,
            "official_api_verified": False,
            "compatible_local_checkpoint": False,
            "smoke_test": "not_run",
            "status": "blocked",
            "reason": "No verified local package/API and compatible checkpoint; no download or external API access attempted.",
        }
    report = {
        "mode": "local_only_no_external_data",
        "checkpoint_files_in_cache": checkpoint_files,
        "candidates": results,
        "claims_allowed": False,
    }
    path = root / "reports/experiments" / output_name
    atomic_write_json(report, path)
    return path
