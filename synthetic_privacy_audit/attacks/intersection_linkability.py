from __future__ import annotations

import numpy as np

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder, nearest_indices
from synthetic_privacy_audit.types import AttackResult


class IntersectionLinkabilityAttack(BaseAttack):
    """Measures records consistently close to multiple independently released tables."""

    name = "intersection_linkability"

    def run(self, context: AttackContext) -> AttackResult:
        self.require_releases(context, 2)
        columns = context.metadata.quasi_identifier_columns or context.metadata.known_columns
        self.require_columns(context, columns)
        encoder = TabularEncoder.fit(context.real_train, columns, context.metadata.categorical_columns)

        def distances_for(targets):
            query = encoder.transform(targets)
            return np.vstack(
                [nearest_indices(query, encoder.transform(release))[1] for release in context.all_releases]
            )

        member_distances = distances_for(context.real_train)
        nonmember_distances = distances_for(context.real_holdout)
        thresholds = np.quantile(nonmember_distances, 0.2, axis=1)
        member_rate = float((member_distances <= thresholds[:, None]).all(axis=0).mean())
        nonmember_rate = float((nonmember_distances <= thresholds[:, None]).all(axis=0).mean())
        return self.completed(
            {
                "release_count": int(len(context.all_releases)),
                "member_intersection_rate": member_rate,
                "nonmember_intersection_rate": nonmember_rate,
                "member_excess": member_rate - nonmember_rate,
                "member_mean_distance": float(member_distances.mean()),
                "nonmember_mean_distance": float(nonmember_distances.mean()),
            },
            details={"persistent_rule": "target is within the closest 20 percent for every release"},
        )
