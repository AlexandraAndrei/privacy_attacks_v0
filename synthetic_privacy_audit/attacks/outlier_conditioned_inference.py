from __future__ import annotations

from dataclasses import replace

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import top_quantile_indices
from synthetic_privacy_audit.attacks.nearest_neighbor_attribute_inference import NearestNeighborAttributeInferenceAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import outlier_scores
from synthetic_privacy_audit.types import AttackResult


class OutlierConditionedInferenceAttack(BaseAttack):
    """Restricts nearest-neighbour attribute inference to the rarest real records."""

    name = "outlier_conditioned_attribute_inference"

    def __init__(self, fraction: float = 0.1) -> None:
        self.fraction = fraction

    def run(self, context: AttackContext) -> AttackResult:
        columns = context.metadata.quasi_identifier_columns or context.metadata.known_columns
        self.require_columns(context, columns)
        scores = outlier_scores(
            context.real_train,
            context.real_holdout,
            columns,
            context.metadata.categorical_columns,
        )
        indices = top_quantile_indices(scores, self.fraction)
        subset_context = replace(context, real_holdout=context.real_holdout.iloc[indices].reset_index(drop=True))
        base = NearestNeighborAttributeInferenceAttack().run(subset_context)
        return self.completed(
            {
                "outlier_fraction": self.fraction,
                "outlier_count": int(len(indices)),
                **base.metrics,
            },
            details={"base_attack": base.attack, "outlier_score": "mixed numeric distance plus inverse category frequency"},
        )

