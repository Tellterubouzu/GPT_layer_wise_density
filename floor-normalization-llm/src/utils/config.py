import json
from typing import Any, Dict, List


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)


def _set_nested(config: Dict[str, Any], keys: List[str], value: Any) -> None:
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override}")
        key, raw_value = override.split("=", 1)
        _set_nested(config, key.split("."), _parse_value(raw_value))
    return config
