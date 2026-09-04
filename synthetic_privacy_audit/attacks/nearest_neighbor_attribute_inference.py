from __future__ import annotations

import numpy as np

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import nearest_sensitive_predictions, sensitive_success
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class NearestNeighborAttributeInferenceAttack(BaseAttack):
    """No-box nearest synthetic neighbour attribute inference with tolerance scoring."""

    name = "nearest_neighbor_attribute_inference"

    def run(self, context: AttackContext) -> AttackResult:
        self.require_columns(context, [*context.metadata.known_columns, *context.metadata.sensitive_columns])
        predictions, distances = nearest_sensitive_predictions(context, context.real_holdout)
        success, per_column = sensitive_success(
            context.real_holdout.reset_index(drop=True),
            predictions,
            context,
        )
        return self.completed(
            {
                "joint_success": float(success.mean()),
                "mean_nearest_distance": float(distances.mean()),
                "p95_nearest_distance": float(np.quantile(distances, 0.95)),
                **per_column,
            },
            details={"scoring": "exact categorical match; configured or 0.1 standard-deviation continuous tolerance"},
        )
