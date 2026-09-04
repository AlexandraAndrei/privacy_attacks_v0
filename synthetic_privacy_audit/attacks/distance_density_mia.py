from __future__ import annotations

import numpy as np
from sklearn.neighbors import KernelDensity

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import membership_distance_scores, safe_auc
from synthetic_privacy_audit.attacks.mia_utils import common_columns, reference_population
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder
from synthetic_privacy_audit.types import AttackResult


class DistanceDensityMembershipInferenceAttack(BaseAttack):
    """CD-H, CD-L and KDE membership signals from one synthetic release."""

    name = "distance_density_membership_inference"

    def run(self, context: AttackContext) -> AttackResult:
        member_l2 = membership_distance_scores(context, context.real_train, metric="euclidean")
        nonmember_l2 = membership_distance_scores(context, context.real_holdout, metric="euclidean")
        member_hamming = membership_distance_scores(context, context.real_train, metric="hamming")
        nonmember_hamming = membership_distance_scores(context, context.real_holdout, metric="hamming")
        reference = reference_population(context)
        columns = common_columns(context)
        encoder = TabularEncoder.fit(reference, columns, context.metadata.categorical_columns)
        synthetic_x = encoder.transform(context.synthetic)
        reference_x = encoder.transform(reference)
        members_x = encoder.transform(context.real_train)
        nonmembers_x = encoder.transform(context.real_holdout)
        synthetic_kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(synthetic_x)
        reference_kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(reference_x)
        member_kde = synthetic_kde.score_samples(members_x) - reference_kde.score_samples(members_x)
        nonmember_kde = synthetic_kde.score_samples(nonmembers_x) - reference_kde.score_samples(nonmembers_x)
        cd_l = safe_auc(member_l2, nonmember_l2)
        cd_h = safe_auc(member_hamming, nonmember_hamming)
        kde = safe_auc(member_kde, nonmember_kde)
        return self.completed(
            {
                "cd_l2_auc": cd_l,
                "cd_hamming_auc": cd_h,
                "kernel_estimator_auc": kde,
                "headline_auc": max(cd_l, cd_h, kde),
            },
            details={"reference_population": "reference.csv, real_control.csv, or real_holdout.csv in that order"},
        )

