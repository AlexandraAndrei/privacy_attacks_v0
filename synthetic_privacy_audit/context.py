from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import pandas as pd


class SyntheticGenerator(Protocol):
    """Adapter for attacks that require retraining a controllable synthesizer."""

    def generate(self, train_data: pd.DataFrame, *, seed: int) -> pd.DataFrame:
        """Fit and generate one synthetic release from train_data."""


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    dataset_name: str
    known_columns: tuple[str, ...]
    sensitive_columns: tuple[str, ...]
    quasi_identifier_columns: tuple[str, ...]
    temporal_column: str | None = None
    categorical_columns: tuple[str, ...] = ()
    continuous_tolerances: dict[str, float] = field(default_factory=dict)
    property_columns: tuple[str, ...] = ()


@dataclass(slots=True)
class AttackContext:
    """All data/resources an attack may consume."""

    metadata: DatasetMetadata
    real_train: pd.DataFrame
    real_holdout: pd.DataFrame
    synthetic: pd.DataFrame
    real_control: pd.DataFrame | None = None
    reference_data: pd.DataFrame | None = None
    synthetic_releases: Sequence[pd.DataFrame] = ()
    generator: SyntheticGenerator | None = None
    seed: int = 42

    def frames_with_columns(self, columns: Sequence[str]) -> list[pd.DataFrame]:
        required = list(columns)
        frames = [self.real_train, self.real_holdout, self.synthetic]
        if self.real_control is not None:
            frames.append(self.real_control)
        if self.reference_data is not None:
            frames.append(self.reference_data)
        for index, frame in enumerate(frames):
            missing = sorted(set(required) - set(frame.columns))
            if missing:
                raise ValueError(f"Frame #{index} is missing required columns: {missing}")
        return frames

    @property
    def all_releases(self) -> tuple[pd.DataFrame, ...]:
        return (self.synthetic, *self.synthetic_releases)

