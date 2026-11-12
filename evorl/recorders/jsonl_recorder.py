"""JSON Lines recording and loading merged records."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from .recorder import Recorder
from .recorder_utils import jsonable, normalize_step, sanitize_filename


class JSONLRecorder(Recorder):
    """Record metadata and metrics to `path/<name>.jsonl`.

    The filename replaces `/` with `_`; metadata retains the original name.
    Initialization overwrites the file and writes a metadata header. Metric
    records remain in write order, allowing out-of-order and duplicate steps.
    Each write flushes the file. Nonfinite numbers and pandas missing values
    become `null`; arrays and Series become lists, and tables retain columns/rows.
    """

    def __init__(self, *, project, name, config, tags, path, group=None):
        self.project = project
        self.name = name
        self.config = config
        self.tags = tags
        self.path = Path(path)
        self.group = group
        self.log_path = self.path / f"{sanitize_filename(name)}.jsonl"
        self._file: TextIO | None = None

    def init(self) -> None:
        if self._file is not None and not self._file.closed:
            raise RuntimeError("JSONLRecorder is already initialized")
        metadata = {
            "project": self.project,
            "name": self.name,
            "group": self.group,
            "config": self.config,
            "tags": self.tags,
        }
        line = json.dumps(
            {"metadata": _json_value(metadata)}, ensure_ascii=False, allow_nan=False
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", encoding="utf-8")
        self._file.write(line + "\n")
        self._file.flush()

    def write(self, data: Mapping[str, Any], step: int) -> None:
        """Append a record with the supplied step, retaining all duplicates."""
        step = normalize_step(step)
        if self._file is None or self._file.closed:
            raise RuntimeError("Initialize JSONLRecorder before writing")
        record = {"step": step, "data": _json_value(data)}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _json_value(value: Any):
    """Convert metrics and nonfinite numbers to standard JSON values."""
    if isinstance(value, pd.DataFrame):
        table = value.to_dict(orient="split")
        value = {"columns": table["columns"], "data": table["data"]}
    elif isinstance(value, pd.Series):
        value = value.tolist()
    elif not isinstance(value, (Mapping, list, tuple)):
        value = jsonable(value)

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load all records, merge equal steps, and return a list sorted by step.

    Nested dictionaries merge by key; later writes replace conflicting values,
    including whole arrays and `null`. Unmentioned metrics remain unchanged.
    The optional first-line metadata header is skipped.
    Invalid or incomplete records raise `ValueError` with their file/line number.
    """
    path = Path(path)
    data_by_step: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
                if (
                    line_number == 1
                    and isinstance(record, dict)
                    and set(record) == {"metadata"}
                ):
                    if not isinstance(record["metadata"], dict):
                        raise TypeError("expected a metadata mapping")
                    continue
                if not isinstance(record, dict) or not isinstance(
                    record.get("data"), dict
                ):
                    raise TypeError("expected an object with a data mapping")
                step = normalize_step(record.get("step"))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            _merge_metrics(data_by_step.setdefault(step, {}), record["data"])
    return [{"step": step, "data": data_by_step[step]} for step in sorted(data_by_step)]


def _merge_metrics(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge nested metric keys, replacing other values with the latest write."""
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_metrics(target[key], value)
        else:
            target[key] = value
