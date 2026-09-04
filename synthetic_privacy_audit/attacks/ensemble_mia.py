from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.attacks.common import safe_auc
from synthetic_privacy_audit.attacks.mia_utils import shadow_membership_signals, target_signals
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class EnsembleMembershipInferenceAttack(BaseAttack):
    """Shadow ensemble combining L2, Hamming and record-rarity signals."""

    name = "ensemble_membership_inference"

    def __init__(self, shadow_runs: int = 8, candidates_per_run: int = 100) -> None:
        self.shadow_runs = shadow_runs
        self.candidates_per_run = candidates_per_run

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise AttackPrerequisiteError("Requires reference_data for ensemble shadow training.")
        l2_score, difficulty, labels = shadow_membership_signals(
            context, shadow_runs=self.shadow_runs, candidates_per_run=self.candidates_per_run
        )
        hamming_score, _, hamming_labels = shadow_membership_signals(
            context,
            shadow_runs=self.shadow_runs,
            candidates_per_run=self.candidates_per_run,
            metric="hamming",
        )
        if not np.array_equal(labels, hamming_labels):
            raise RuntimeError("Shadow labels are not aligned across ensemble signals.")
        l2_target, target_difficulty, target_labels = target_signals(context)
        hamming_target, _, hamming_target_labels = target_signals(context, metric="hamming")
        if not np.array_equal(target_labels, hamming_target_labels):
            raise RuntimeError("Target labels are not aligned across ensemble signals.")
        features = np.column_stack([l2_score, hamming_score, difficulty])
        classifier = RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=context.seed, n_jobs=-1
        ).fit(features, labels)
        probability = classifier.predict_proba(
            np.column_stack([l2_target, hamming_target, target_difficulty])
        )[:, 1]
        return self.completed(
            {
                "ensemble_auc": safe_auc(probability[target_labels == 1], probability[target_labels == 0]),
                "shadow_examples": int(len(labels)),
                "feature_count": 3,
            },
            details={"signals": ["closest-distance-l2", "closest-distance-hamming", "outlier-rarity"]},
        )

