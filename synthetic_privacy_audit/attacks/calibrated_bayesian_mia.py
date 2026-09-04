from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.attacks.common import safe_auc
from synthetic_privacy_audit.attacks.mia_utils import shadow_membership_signals, target_signals
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class CalibratedBayesianMembershipInferenceAttack(BaseAttack):
    """Difficulty-calibrated posterior membership estimates trained on shadows."""

    name = "calibrated_bayesian_membership_inference"

    def __init__(self, shadow_runs: int = 10, candidates_per_run: int = 100) -> None:
        self.shadow_runs = shadow_runs
        self.candidates_per_run = candidates_per_run

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise AttackPrerequisiteError("Requires reference_data for calibrated shadow membership inference.")
        shadow_score, shadow_difficulty, shadow_labels = shadow_membership_signals(
            context,
            shadow_runs=self.shadow_runs,
            candidates_per_run=self.candidates_per_run,
        )
        target_score, target_difficulty, target_labels = target_signals(context)
        scaler = StandardScaler().fit(np.column_stack([shadow_score, shadow_difficulty]))
        classifier = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=context.seed)
        classifier.fit(scaler.transform(np.column_stack([shadow_score, shadow_difficulty])), shadow_labels)
        posterior = classifier.predict_proba(
            scaler.transform(np.column_stack([target_score, target_difficulty]))
        )[:, 1]
        calibrated_auc = safe_auc(posterior[target_labels == 1], posterior[target_labels == 0])
        uncalibrated_auc = safe_auc(
            target_score[target_labels == 1], target_score[target_labels == 0]
        )
        return self.completed(
            {
                "calibrated_posterior_auc": calibrated_auc,
                "uncalibrated_distance_auc": uncalibrated_auc,
                "calibration_delta_auc": calibrated_auc - uncalibrated_auc,
            },
            details={"calibration_features": ["closest synthetic distance", "outlier difficulty"]},
        )

