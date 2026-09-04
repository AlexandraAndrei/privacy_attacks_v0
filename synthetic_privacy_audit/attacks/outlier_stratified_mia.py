from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import membership_distance_scores, safe_auc
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import outlier_scores
from synthetic_privacy_audit.types import AttackResult


class OutlierStratifiedMembershipInferenceAttack(BaseAttack):
    """Reports closest-distance MIA performance in ten real-data rarity buckets."""

    name = "outlier_decile_stratified_mia"

    def run(self, context: AttackContext) -> AttackResult:
        columns = context.metadata.quasi_identifier_columns or context.metadata.known_columns
        self.require_columns(context, columns)
        members = context.real_train
        nonmembers = context.real_holdout
        member_scores = membership_distance_scores(context, members, metric="euclidean")
        nonmember_scores = membership_distance_scores(context, nonmembers, metric="euclidean")
        member_outlier = outlier_scores(members, members, columns, context.metadata.categorical_columns)
        nonmember_outlier = outlier_scores(members, nonmembers, columns, context.metadata.categorical_columns)
        all_outlier = np.concatenate([member_outlier, nonmember_outlier])
        boundaries = np.quantile(all_outlier, np.linspace(0, 1, 11))
        metrics: dict[str, float | int] = {"overall_auc": safe_auc(member_scores, nonmember_scores)}
        for decile in range(10):
            lower, upper = boundaries[decile], boundaries[decile + 1]
            member_mask = (member_outlier >= lower) & (
                member_outlier <= upper if decile == 9 else member_outlier < upper
            )
            nonmember_mask = (nonmember_outlier >= lower) & (
                nonmember_outlier <= upper if decile == 9 else nonmember_outlier < upper
            )
            if member_mask.any() and nonmember_mask.any():
                metrics[f"decile_{decile + 1}_auc"] = safe_auc(
                    member_scores[member_mask], nonmember_scores[nonmember_mask]
                )
                metrics[f"decile_{decile + 1}_n"] = int(member_mask.sum() + nonmember_mask.sum())
        return self.completed(metrics, details={"decile_10": "rarest records; higher AUC indicates more MIA risk"})

