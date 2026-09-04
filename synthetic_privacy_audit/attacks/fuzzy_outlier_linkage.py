from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import top_quantile_indices
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import outlier_scores
from synthetic_privacy_audit.features.tabular import TabularEncoder
from synthetic_privacy_audit.types import AttackResult


class FuzzyOutlierLinkageAttack(BaseAttack):
    """Worst-case fuzzy linkage for the rarest quasi-identifier records."""

    name = "fuzzy_outlier_reidentification"

    def __init__(self, fraction: float = 0.1, maximum_targets: int = 2_000) -> None:
        self.fraction = fraction
        self.maximum_targets = maximum_targets

    def run(self, context: AttackContext) -> AttackResult:
        qis = context.metadata.quasi_identifier_columns
        self.require_columns(context, qis)
        if len(qis) < 1:
            raise ValueError("Fuzzy linkage requires quasi_identifier_columns.")
        scores = outlier_scores(context.real_train, context.real_holdout, qis, context.metadata.categorical_columns)
        selected = top_quantile_indices(scores, self.fraction)[-self.maximum_targets :]
        targets = context.real_holdout.iloc[selected]
        encoder = TabularEncoder.fit(context.real_train, qis, context.metadata.categorical_columns)
        neighbours = NearestNeighbors(n_neighbors=min(2, len(context.synthetic)), metric="euclidean")
        neighbours.fit(encoder.transform(context.synthetic))
        distances, _ = neighbours.kneighbors(encoder.transform(targets))
        if distances.shape[1] == 1:
            unique = np.ones(len(distances), dtype=bool)
            margins = np.full(len(distances), np.inf)
        else:
            margins = distances[:, 1] - distances[:, 0]
            unique = margins > np.maximum(0.05, 0.15 * distances[:, 0])
        return self.completed(
            {
                "outlier_targets": int(len(targets)),
                "unique_fuzzy_match_rate": float(unique.mean()),
                "mean_best_match_distance": float(distances[:, 0].mean()),
                "mean_match_margin": float(np.mean(margins[np.isfinite(margins)])) if np.isfinite(margins).any() else None,
            },
            details={"success_rule": "nearest match separated from the second match by 15 percent or 0.05"},
        )

