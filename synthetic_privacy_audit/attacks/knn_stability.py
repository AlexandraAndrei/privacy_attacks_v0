from __future__ import annotations

import numpy as np

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder, nearest_indices
from synthetic_privacy_audit.types import AttackResult


class KNNStabilityAttack(BaseAttack):
    """Measures stability of closest synthetic-neighbour distances across releases."""

    name = "knn_stability"

    def run(self, context: AttackContext) -> AttackResult:
        self.require_releases(context, 2)
        columns = context.metadata.quasi_identifier_columns or context.metadata.known_columns
        self.require_columns(context, columns)
        encoder = TabularEncoder.fit(context.real_train, columns, context.metadata.categorical_columns)
        queries = encoder.transform(context.real_holdout)
        distance_matrix = []
        for release in context.all_releases:
            _, distances = nearest_indices(queries, encoder.transform(release))
            distance_matrix.append(distances)
        distances = np.asarray(distance_matrix)
        relative_std = distances.std(axis=0) / np.maximum(distances.mean(axis=0), 1e-9)
        correlations = [
            np.corrcoef(distances[0], distances[index])[0, 1]
            for index in range(1, len(distances))
            if len(distances[0]) > 1
        ]
        return self.completed(
            {
                "release_count": int(len(distances)),
                "mean_relative_distance_std": float(np.nanmean(relative_std)),
                "stable_target_rate": float((relative_std <= 0.1).mean()),
                "mean_distance_correlation": float(np.nanmean(correlations)),
            },
            details={"stable_target_rule": "nearest-neighbour distance changes by at most 10 percent across releases"},
        )

