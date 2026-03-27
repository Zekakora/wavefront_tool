from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from .wavefront_param_schema import ALGORITHM_DEFINITIONS
except ImportError:
    from wavefront_param_schema import ALGORITHM_DEFINITIONS


class ParameterStore:
    def __init__(self, storage_path: str | Path | None = None) -> None:
        if storage_path is None:
            storage_path = Path.home() / ".wavefront_locator" / "user_params.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._defaults = {
            algo_id: deepcopy(info["defaults"])
            for algo_id, info in ALGORITHM_DEFINITIONS.items()
        }
        self._data = {
            algo_id: deepcopy(info["defaults"])
            for algo_id, info in ALGORITHM_DEFINITIONS.items()
        }
        self.load()

    def load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        for algo_id, defaults in self._defaults.items():
            incoming = payload.get(algo_id, {})
            if not isinstance(incoming, dict):
                continue
            merged = deepcopy(defaults)
            for key in defaults:
                if key in incoming:
                    merged[key] = incoming[key]
            self._data[algo_id] = merged

    def save(self) -> None:
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def algorithm_defaults(self, algorithm_id: str) -> dict[str, Any]:
        return deepcopy(self._defaults[algorithm_id])

    def get_params(self, algorithm_id: str) -> dict[str, Any]:
        return deepcopy(self._data[algorithm_id])

    def set_param(self, algorithm_id: str, key: str, value: Any, *, save: bool = True) -> None:
        if algorithm_id not in self._data:
            raise KeyError(f"Unknown algorithm id: {algorithm_id}")
        if key not in self._defaults[algorithm_id]:
            raise KeyError(f"Unknown parameter: {algorithm_id}.{key}")
        self._data[algorithm_id][key] = value
        if save:
            self.save()

    def update_params(self, algorithm_id: str, values: dict[str, Any], *, save: bool = True) -> None:
        if algorithm_id not in self._data:
            raise KeyError(f"Unknown algorithm id: {algorithm_id}")
        for key, value in values.items():
            if key in self._defaults[algorithm_id]:
                self._data[algorithm_id][key] = value
        if save:
            self.save()

    def reset_algorithm(self, algorithm_id: str, *, save: bool = True) -> dict[str, Any]:
        self._data[algorithm_id] = deepcopy(self._defaults[algorithm_id])
        if save:
            self.save()
        return self.get_params(algorithm_id)

    def import_algorithm_json(self, algorithm_id: str, file_path: str | Path, *, save: bool = True) -> dict[str, Any]:
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("导入文件必须是 JSON 对象")

        if algorithm_id in payload and isinstance(payload[algorithm_id], dict):
            payload = payload[algorithm_id]

        merged = self.get_params(algorithm_id)
        for key in self._defaults[algorithm_id]:
            if key in payload:
                merged[key] = payload[key]
        self._data[algorithm_id] = merged
        if save:
            self.save()
        return self.get_params(algorithm_id)

    def export_algorithm_json(self, algorithm_id: str, file_path: str | Path) -> str:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm_id": algorithm_id,
            "algorithm_label": ALGORITHM_DEFINITIONS[algorithm_id]["label"],
            "params": self._data[algorithm_id],
        }
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(file_path)


__all__ = ["ParameterStore"]
