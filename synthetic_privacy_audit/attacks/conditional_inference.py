from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


def _quantile_edges(reference: pd.DataFrame, columns: tuple[str, ...], bins: int = 8) -> dict[str, np.ndarray]:
    edges: dict[str, np.ndarray] = {}
    for column in columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            values = pd.to_numeric(reference[column], errors="coerce").dropna()
            interior = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))[1:-1])
            edges[column] = np.concatenate(([-np.inf], interior, [np.inf]))
    return edges


def _encoded_values(frame: pd.DataFrame, column: str, edges: dict[str, np.ndarray]) -> pd.Series:
    if column not in edges:
        return frame[column].astype("string").fillna("<missing>")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return pd.cut(numeric, bins=edges[column], labels=False, include_lowest=True).fillna(-1).astype("string")


def _conditional_tvd(
    left: pd.DataFrame,
    right: pd.DataFrame,
    known: tuple[str, ...],
    secret: str,
    edges: dict[str, np.ndarray],
) -> tuple[float, int]:
    left_key = pd.concat([_encoded_values(left, column, edges) for column in known], axis=1).agg("|".join, axis=1)
    right_key = pd.concat([_encoded_values(right, column, edges) for column in known], axis=1).agg("|".join, axis=1)
    left_table = pd.crosstab(left_key, _encoded_values(left, secret, edges), normalize="index")
    right_table = pd.crosstab(right_key, _encoded_values(right, secret, edges), normalize="index")
    shared = left_table.index.intersection(right_table.index)
    if shared.empty:
        return float("nan"), 0
    columns = left_table.columns.union(right_table.columns)
    aligned_left = left_table.reindex(index=shared, columns=columns, fill_value=0.0)
    aligned_right = right_table.reindex(index=shared, columns=columns, fill_value=0.0)
    return float(0.5 * np.abs(aligned_left - aligned_right).sum(axis=1).mean()), int(len(shared))


class ConditionalInferenceAttack(BaseAttack):
    """Compares P(sensitive | known attributes) from releases and real data."""

    name = "conditional_inference"

    def run(self, context: AttackContext) -> AttackResult:
        known = tuple(context.metadata.known_columns[: min(3, len(context.metadata.known_columns))])
        self.require_columns(context, [*known, *context.metadata.sensitive_columns])
        if not known:
            raise ValueError("Conditional inference needs known columns.")
        edges = _quantile_edges(
            context.real_train,
            tuple(dict.fromkeys([*known, *context.metadata.sensitive_columns])),
        )
        train_scores = []
        holdout_scores = []
        common_groups = []
        for secret in context.metadata.sensitive_columns:
            train_score, train_groups = _conditional_tvd(
                context.synthetic, context.real_train, known, secret, edges
            )
            holdout_score, holdout_groups = _conditional_tvd(
                context.synthetic, context.real_holdout, known, secret, edges
            )
            train_scores.append(train_score)
            holdout_scores.append(holdout_score)
            common_groups.append(min(train_groups, holdout_groups))
        return self.completed(
            {
                "synthetic_vs_train_tvd": float(np.nanmean(train_scores)),
                "synthetic_vs_holdout_tvd": float(np.nanmean(holdout_scores)),
                "shared_condition_groups": int(min(common_groups, default=0)),
            },
            details={"lower_tvd_means": "synthetic conditionals more closely reproduce real conditionals"},
        )

