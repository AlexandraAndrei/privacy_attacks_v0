from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import is_continuous, sensitive_success
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder
from synthetic_privacy_audit.types import AttackResult


class LinearReconstructionAttributeInferenceAttack(BaseAttack):
    """Linear no-box reconstruction using synthetic marginal/conditional features."""

    name = "linear_reconstruction_attribute_inference"

    def run(self, context: AttackContext) -> AttackResult:
        known = context.metadata.known_columns
        sensitive = context.metadata.sensitive_columns
        self.require_columns(context, [*known, *sensitive])
        encoder = TabularEncoder.fit(context.synthetic, known, context.metadata.categorical_columns)
        synthetic_x = encoder.transform(context.synthetic)
        holdout_x = encoder.transform(context.real_holdout)
        predictions: dict[str, object] = {}
        for column in sensitive:
            if is_continuous(context.synthetic, column):
                target = pd.to_numeric(context.synthetic[column], errors="coerce").fillna(0.0)
                model = Ridge(alpha=1.0).fit(synthetic_x, target)
                predictions[column] = model.predict(holdout_x)
            else:
                labels = LabelEncoder()
                target = labels.fit_transform(context.synthetic[column].astype("string").fillna("<missing>"))
                model = LogisticRegression(max_iter=500, n_jobs=None).fit(synthetic_x, target)
                predictions[column] = labels.inverse_transform(model.predict(holdout_x))
        predicted = pd.DataFrame(predictions)
        success, per_column = sensitive_success(
            context.real_holdout.reset_index(drop=True),
            predicted.reset_index(drop=True),
            context,
        )
        return self.completed(
            {"joint_success": float(success.mean()), **per_column},
            details={"solver": "regularized linear/logistic reconstruction; replaceable by Gurobi LP adapter"},
        )

