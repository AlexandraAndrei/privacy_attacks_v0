from __future__ import annotations

import pandas as pd

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import fit_sensitive_models, predict_sensitive, sensitive_success
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class ShadowAttributeInferenceAttack(BaseAttack):
    """Groundhog-style shadow-release attribute inference scaffold."""

    name = "shadow_attribute_inference"

    def __init__(self, shadow_runs: int = 5) -> None:
        self.shadow_runs = shadow_runs

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise ValueError("Requires reference_data for shadow generator training.")
        self.require_columns(context, [*context.metadata.known_columns, *context.metadata.sensitive_columns])
        releases = []
        for shadow_id in range(self.shadow_runs):
            sample = context.reference_data.sample(
                frac=0.5,
                replace=False,
                random_state=context.seed + shadow_id,
            )
            releases.append(context.generator.generate(sample, seed=context.seed + shadow_id))
        shadow_synthetic = pd.concat(releases, ignore_index=True)
        models = fit_sensitive_models(
            shadow_synthetic,
            context.metadata.known_columns,
            context.metadata.sensitive_columns,
            context.metadata.categorical_columns,
            seed=context.seed,
        )
        prediction = predict_sensitive(models, context.real_holdout)
        success, per_column = sensitive_success(
            context.real_holdout.reset_index(drop=True), prediction.reset_index(drop=True), context
        )
        return self.completed(
            {"joint_success": float(success.mean()), "shadow_runs": self.shadow_runs, **per_column},
            details={"threat_model": "no-box shadow releases trained on reference population"},
        )

