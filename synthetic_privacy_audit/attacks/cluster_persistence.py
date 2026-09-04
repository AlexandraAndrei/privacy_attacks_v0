from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder
from synthetic_privacy_audit.types import AttackResult


class ClusterPersistenceAttack(BaseAttack):
    """Matches synthetic cluster signatures across release pairs."""

    name = "cluster_persistence"

    def __init__(self, max_clusters: int = 20) -> None:
        self.max_clusters = max_clusters

    def run(self, context: AttackContext) -> AttackResult:
        self.require_releases(context, 2)
        columns = context.metadata.quasi_identifier_columns or context.metadata.known_columns
        self.require_columns(context, columns)
        encoder = TabularEncoder.fit(context.real_train, columns, context.metadata.categorical_columns)
        release_arrays = [encoder.transform(release) for release in context.all_releases]
        minimum_rows = min(len(array) for array in release_arrays)
        clusters = max(2, min(self.max_clusters, int(np.sqrt(minimum_rows))))
        centres = [
            MiniBatchKMeans(n_clusters=clusters, random_state=context.seed + index, n_init="auto")
            .fit(array)
            .cluster_centers_
            for index, array in enumerate(release_arrays)
        ]
        similarities = []
        for left, right in zip(centres, centres[1:]):
            distances = np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2))
            rows, cols = linear_sum_assignment(distances)
            similarities.append(float(np.exp(-distances[rows, cols]).mean()))
        return self.completed(
            {
                "release_count": int(len(centres)),
                "clusters_per_release": clusters,
                "mean_matched_cluster_similarity": float(np.mean(similarities)),
            },
            details={"similarity": "exponential of matched centroid distance; higher is more persistent"},
        )

