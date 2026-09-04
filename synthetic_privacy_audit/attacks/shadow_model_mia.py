from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.attacks.common import safe_auc
from synthetic_privacy_audit.attacks.mia_utils import shadow_membership_signals, target_signals
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class ShadowModelMembershipInferenceAttack(BaseAttack):
    """NaiveGroundhog/HistGroundhog-style black-box shadow membership inference."""

    name = "shadow_model_membership_inference"

    def __init__(self, shadow_runs: int = 8, candidates_per_run: int = 100) -> None:
        self.shadow_runs = shadow_runs
        self.candidates_per_run = candidates_per_run

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise AttackPrerequisiteError("Requires reference_data for NaiveGroundhog/HistGroundhog shadows.")
        shadow_score, shadow_difficulty, labels = shadow_membership_signals(
            context,
            shadow_runs=self.shadow_runs,
            candidates_per_run=self.candidates_per_run,
        )
        target_score, target_difficulty, target_labels = target_signals(context)
        naive = RandomForestClassifier(n_estimators=300, random_state=context.seed, n_jobs=-1)
        naive.fit(shadow_score.reshape(-1, 1), labels)
        histogram = RandomForestClassifier(n_estimators=300, random_state=context.seed + 1, n_jobs=-1)
        histogram.fit(np.column_stack([shadow_score, shadow_difficulty]), labels)
        naive_auc = safe_auc(
            naive.predict_proba(target_score.reshape(-1, 1))[:, 1][target_labels == 1],
            naive.predict_proba(target_score.reshape(-1, 1))[:, 1][target_labels == 0],
        )
        hist_probabilities = histogram.predict_proba(
            np.column_stack([target_score, target_difficulty])
        )[:, 1]
        hist_auc = safe_auc(hist_probabilities[target_labels == 1], hist_probabilities[target_labels == 0])
        return self.completed(
            {
                "naive_groundhog_auc": naive_auc,
                "hist_groundhog_auc": hist_auc,
                "headline_auc": max(naive_auc, hist_auc),
                "shadow_examples": int(len(labels)),
            },
            details={"headline": "maximum AUC across the two black-box shadow variants"},
        )

