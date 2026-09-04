from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import fit_sensitive_models, is_continuous
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class TopKAttributeInferenceAttack(BaseAttack):
    """Top-k success for categorical secrets and tolerance success for continuous ones."""

    name = "top_k_attribute_inference"

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def run(self, context: AttackContext) -> AttackResult:
        self.require_columns(context, [*context.metadata.known_columns, *context.metadata.sensitive_columns])
        models = fit_sensitive_models(
            context.synthetic,
            context.metadata.known_columns,
            context.metadata.sensitive_columns,
            context.metadata.categorical_columns,
            seed=context.seed,
        )
        features = models.encoder.transform(context.real_holdout)
        scores: dict[str, float] = {}
        for column, (model, labels, continuous) in models.models.items():
            if continuous:
                prediction = model.predict(features)
                actual = pd.to_numeric(context.real_holdout[column], errors="coerce").to_numpy(dtype=float)
                tolerance = context.metadata.continuous_tolerances.get(
                    column,
                    max(float(pd.to_numeric(context.real_holdout[column], errors="coerce").std(ddof=0)) * 0.1, 1e-9),
                )
                success = np.abs(actual - prediction) <= tolerance
            else:
                probabilities = model.predict_proba(features)
                top = np.argsort(probabilities, axis=1)[:, -min(self.k, probabilities.shape[1]):]
                actual = labels.transform(context.real_holdout[column].astype("string").fillna("<missing>"))
                success = np.array([value in choices for value, choices in zip(actual, top)])
            scores[f"{column}_top_{self.k}_success"] = float(np.mean(success))
        return self.completed(
            {"mean_top_k_success": float(np.mean(list(scores.values()))), "k": self.k, **scores},
            details={"continuous_columns": "scored with configured tolerance rather than ranked classes"},
        )

