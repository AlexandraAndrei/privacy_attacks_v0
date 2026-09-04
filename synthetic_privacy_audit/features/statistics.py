from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def outlier_scores(
    reference: pd.DataFrame,
    targets: pd.DataFrame,
    columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
) -> np.ndarray:
    categorical = set(categorical_columns)
    scores = np.zeros(len(targets), dtype=float)
    numeric = [column for column in columns if column not in categorical]
    for column in numeric:
        values = pd.to_numeric(reference[column], errors="coerce")
        scale = max(float(values.std(ddof=0)), 1e-9)
        scores += np.abs(pd.to_numeric(targets[column], errors="coerce").fillna(values.median()) - values.mean()) / scale
    for column in columns:
        if column in categorical or not pd.api.types.is_numeric_dtype(reference[column]):
            probabilities = reference[column].astype("string").value_counts(normalize=True)
            p = targets[column].astype("string").map(probabilities).fillna(1.0 / (len(reference) + 1))
            scores += -np.log(np.maximum(p.to_numpy(dtype=float), 1.0 / (len(reference) + 1)))
    return scores


def empirical_conditional(
    frame: pd.DataFrame,
    known_columns: Sequence[str],
    sensitive_column: str,
    max_groups: int = 100,
) -> dict[str, np.ndarray]:
    if not known_columns:
        raise ValueError("Conditional inference needs at least one known column.")
    key = frame.loc[:, list(known_columns)].astype("string").agg("|".join, axis=1)
    labels = frame[sensitive_column].astype("string")
    grouped = pd.crosstab(key, labels, normalize="index").head(max_groups)
    return {str(index): row.to_numpy(dtype=float) for index, row in grouped.iterrows()}


def dataset_signature(
    frame: pd.DataFrame,
    columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
    bins: int = 8,
) -> np.ndarray:
    categorical = set(categorical_columns)
    values: list[float] = []
    for column in columns:
        if column in categorical or not pd.api.types.is_numeric_dtype(frame[column]):
            frequencies = frame[column].astype("string").value_counts(normalize=True).head(bins)
            values.extend(frequencies.to_list())
            values.extend([0.0] * (bins - len(frequencies)))
        else:
            numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
            if numeric.empty:
                values.extend([0.0] * (bins + 2))
            else:
                values.extend([float(numeric.mean()), float(numeric.std(ddof=0))])
                values.extend(np.quantile(numeric, np.linspace(0, 1, bins)).tolist())
    return np.asarray(values, dtype=float)

