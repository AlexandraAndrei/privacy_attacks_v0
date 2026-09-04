from __future__ import annotations

import numpy as np
from sklearn.neighbors import KernelDensity

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.attacks.common import safe_auc
from synthetic_privacy_audit.attacks.mia_utils import shadow_membership_signals, target_signals
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class ShadowLikelihoodRatioMembershipInferenceAttack(BaseAttack):
    """Shadow-model likelihood-ratio MIA over closest-synthetic-distance signals."""

    name = "shadow_likelihood_ratio_membership_inference"

    def __init__(self, shadow_runs: int = 12, candidates_per_run: int = 100) -> None:
        self.shadow_runs = shadow_runs
        self.candidates_per_run = candidates_per_run

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise AttackPrerequisiteError("Requires reference_data and shadow generators for likelihood ratios.")
        scores, _, labels = shadow_membership_signals(
            context, shadow_runs=self.shadow_runs, candidates_per_run=self.candidates_per_run
        )
        in_kde = KernelDensity(bandwidth=0.2).fit(scores[labels == 1].reshape(-1, 1))
        out_kde = KernelDensity(bandwidth=0.2).fit(scores[labels == 0].reshape(-1, 1))
        target_scores, _, target_labels = target_signals(context)
        log_ratio = (
            in_kde.score_samples(target_scores.reshape(-1, 1))
            - out_kde.score_samples(target_scores.reshape(-1, 1))
        )
        return self.completed(
            {
                "likelihood_ratio_auc": safe_auc(log_ratio[target_labels == 1], log_ratio[target_labels == 0]),
                "shadow_examples": int(len(labels)),
            },
            details={"calibration": "KDE likelihood under member versus non-member shadow signals"},
        )

