from __future__ import annotations

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import fit_sensitive_models, predict_sensitive, sensitive_success
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class JointAttributeInferenceAttack(BaseAttack):
    """Predict several sensitive attributes jointly from known attributes."""

    name = "joint_attribute_inference"

    def run(self, context: AttackContext) -> AttackResult:
        self.require_columns(context, [*context.metadata.known_columns, *context.metadata.sensitive_columns])
        models = fit_sensitive_models(
            context.synthetic,
            context.metadata.known_columns,
            context.metadata.sensitive_columns,
            context.metadata.categorical_columns,
            seed=context.seed,
        )
        predictions = predict_sensitive(models, context.real_holdout)
        joint_success, per_column = sensitive_success(
            context.real_holdout.reset_index(drop=True),
            predictions.reset_index(drop=True),
            context,
        )
        return self.completed(
            {
                "joint_exact_success": float(joint_success.mean()),
                "evaluated_records": int(len(joint_success)),
                **per_column,
            },
            details={"training_source": "target synthetic release", "evaluation_source": "real holdout"},
        )

