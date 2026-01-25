from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DATE_KEYS = {
    "fromdate",
    "todate",
    "backtest_start",
    "backtest_end",
    "start_date",
    "end_date",
}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str, base_dir: Optional[Path] = None) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    return str(path)


def _parse_datetime(value: str) -> datetime | str:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return value


def _coerce_dates(obj: Any, date_keys: Iterable[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in date_keys and isinstance(value, str):
                obj[key] = _parse_datetime(value)
            else:
                obj[key] = _coerce_dates(value, date_keys)
        return obj
    if isinstance(obj, list):
        return [_coerce_dates(item, date_keys) for item in obj]
    return obj


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML configs. Install with: pip install pyyaml"
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)

    return _coerce_dates(config, DATE_KEYS)
