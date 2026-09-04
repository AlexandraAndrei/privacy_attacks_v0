from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_privacy_audit.context import AttackContext, DatasetMetadata


def _read_csv(path_value: str | None, base_dir: Path, *, required: bool) -> pd.DataFrame | None:
    if not path_value:
        if required:
            raise ValueError("Required CSV path is missing from the configuration.")
        return None
    path = (base_dir / path_value).resolve()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required input is missing: {path}")
        return None
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def load_context(config_path: str | Path) -> AttackContext:
    path = Path(config_path).resolve()
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    metadata = DatasetMetadata(
        dataset_name=raw["dataset_name"],
        known_columns=tuple(raw["known_columns"]),
        sensitive_columns=tuple(raw["sensitive_columns"]),
        quasi_identifier_columns=tuple(raw["quasi_identifier_columns"]),
        temporal_column=raw.get("temporal_column"),
        categorical_columns=tuple(raw.get("categorical_columns", [])),
        continuous_tolerances=dict(raw.get("continuous_tolerances", {})),
        property_columns=tuple(raw.get("property_columns", raw["sensitive_columns"])),
    )
    base_dir = path.parent.parent
    releases = [
        _read_csv(item, base_dir, required=True)
        for item in raw.get("synthetic_releases", [])
    ]
    return AttackContext(
        metadata=metadata,
        real_train=_read_csv(raw.get("real_train"), base_dir, required=True),
        real_holdout=_read_csv(raw.get("real_holdout"), base_dir, required=True),
        synthetic=_read_csv(raw.get("synthetic"), base_dir, required=True),
        real_control=_read_csv(raw.get("real_control"), base_dir, required=False),
        reference_data=_read_csv(raw.get("reference_data"), base_dir, required=False),
        synthetic_releases=tuple(frame for frame in releases if frame is not None),
        seed=int(raw.get("seed", 42)),
    )

