from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.attacks.shadow_property_inference import _property_indicator, _resample_property
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import dataset_signature
from synthetic_privacy_audit.types import AttackResult


class PropertyMetaClassifierAttack(BaseAttack):
    """Maps synthetic dataset signatures to a continuous hidden population property."""

    name = "property_meta_classifier"

    def __init__(self, shadow_runs: int = 12) -> None:
        self.shadow_runs = shadow_runs

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise AttackPrerequisiteError("Requires reference_data for property meta-classifier.")
        columns = context.metadata.property_columns
        self.require_columns(context, columns)
        property_column = columns[0]
        signatures, prevalences = [], []
        size = min(len(context.reference_data), len(context.real_train))
        for run in range(self.shadow_runs):
            high = bool(run % 2)
            subset = _resample_property(
                context.reference_data, property_column, high, size, context.seed + run * 7
            )
            release = context.generator.generate(subset, seed=context.seed + run)
            signatures.append(dataset_signature(release, columns, context.metadata.categorical_columns))
            prevalences.append(float(_property_indicator(subset, property_column).mean()))
        model = RandomForestRegressor(n_estimators=300, random_state=context.seed, n_jobs=-1)
        model.fit(np.vstack(signatures), prevalences)
        target_signature = dataset_signature(context.synthetic, columns, context.metadata.categorical_columns)
        estimate = float(model.predict(target_signature.reshape(1, -1))[0])
        actual = float(_property_indicator(context.real_train, property_column).mean())
        return self.completed(
            {
                "estimated_train_property_prevalence": estimate,
                "actual_train_property_prevalence": actual,
                "absolute_error": abs(estimate - actual),
                "shadow_runs": self.shadow_runs,
            },
            details={"property_column": property_column, "model": "black-box random-forest meta-regressor"},
        )

