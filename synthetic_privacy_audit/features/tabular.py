from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularEncoder:
    """A deterministic mixed-type encoder shared by distance and KDE attacks."""

    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_medians: dict[str, float]
    scaler: StandardScaler | None
    dummy_columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        columns: Sequence[str],
        categorical_columns: Sequence[str] = (),
    ) -> "TabularEncoder":
        columns = tuple(columns)
        specified = set(categorical_columns)
        categorical = tuple(
            column
            for column in columns
            if column in specified or not pd.api.types.is_numeric_dtype(frame[column])
        )
        numeric = tuple(column for column in columns if column not in categorical)
        medians = {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in numeric
        }
        prepared = cls._prepare(frame, columns, categorical, medians)
        scaler = StandardScaler().fit(prepared.loc[:, list(numeric)]) if numeric else None
        dummies = tuple(column for column in prepared.columns if column not in numeric)
        return cls(columns, categorical, medians, scaler, dummies)

    @staticmethod
    def _prepare(
        frame: pd.DataFrame,
        columns: Sequence[str],
        categorical: Sequence[str],
        medians: dict[str, float],
    ) -> pd.DataFrame:
        subset = frame.loc[:, list(columns)].copy()
        for column, median in medians.items():
            subset[column] = (
                pd.to_numeric(subset[column], errors="coerce").fillna(median).astype(float)
            )
        for column in categorical:
            subset[column] = subset[column].astype("string").fillna("<missing>")
        return pd.get_dummies(subset, columns=list(categorical), dtype=float)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = self._prepare(frame, self.columns, self.categorical_columns, self.numeric_medians)
        numeric = [column for column in self.columns if column not in self.categorical_columns]
        for column in self.dummy_columns:
            if column not in prepared:
                prepared[column] = 0.0
        prepared = prepared.reindex(columns=[*numeric, *self.dummy_columns], fill_value=0.0)
        if numeric and self.scaler is not None:
            prepared.loc[:, numeric] = self.scaler.transform(prepared.loc[:, numeric])
        return prepared.to_numpy(dtype=float)


def nearest_indices(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    if not len(candidates):
        raise ValueError("Candidate synthetic release is empty.")
    neighbours = NearestNeighbors(n_neighbors=1, metric=metric)
    neighbours.fit(candidates)
    distances, indices = neighbours.kneighbors(query)
    return indices[:, 0], distances[:, 0]


def mixed_feature_columns(frame: pd.DataFrame, excluded: Sequence[str] = ()) -> tuple[str, ...]:
    excluded_set = set(excluded)
    columns = tuple(column for column in frame.columns if column not in excluded_set)
    if not columns:
        raise ValueError("No usable feature columns remain.")
    return columns
