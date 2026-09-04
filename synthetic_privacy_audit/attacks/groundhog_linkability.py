from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.attacks.common import safe_auc
from synthetic_privacy_audit.attacks.mia_utils import common_columns
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import dataset_signature
from synthetic_privacy_audit.features.tabular import TabularEncoder
from synthetic_privacy_audit.types import AttackResult


class GroundhogLinkabilityAttack(BaseAttack):
    """Groundhog-Day linkability with F_Naive and F_Hist release signatures."""

    name = "groundhog_linkability"

    def __init__(self, shadow_runs: int = 8, candidates_per_run: int = 75) -> None:
        self.shadow_runs = shadow_runs
        self.candidates_per_run = candidates_per_run

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise AttackPrerequisiteError("Requires reference_data for Groundhog linkability shadows.")
        columns = common_columns(context)
        reference = context.reference_data.reset_index(drop=True)
        encoder = TabularEncoder.fit(reference, columns, context.metadata.categorical_columns)
        naive_features, histogram_features, labels = [], [], []
        for run in range(self.shadow_runs):
            members = reference.sample(frac=0.5, random_state=context.seed + run)
            release = context.generator.generate(members, seed=context.seed + run)
            signature = dataset_signature(release, columns, context.metadata.categorical_columns)
            candidates = reference.sample(
                n=min(self.candidates_per_run, len(reference)),
                random_state=context.seed + 100 + run,
            )
            candidate_features = encoder.transform(candidates)
            membership = candidates.index.isin(members.index).astype(int)
            naive_signature = signature[::2]
            naive_features.append(
                np.hstack([candidate_features, np.tile(naive_signature, (len(candidates), 1))])
            )
            histogram_features.append(
                np.hstack([candidate_features, np.tile(signature, (len(candidates), 1))])
            )
            labels.extend(membership)
        labels_array = np.asarray(labels)
        naive = RandomForestClassifier(n_estimators=300, random_state=context.seed, n_jobs=-1).fit(
            np.vstack(naive_features), labels_array
        )
        histogram = RandomForestClassifier(n_estimators=300, random_state=context.seed + 1, n_jobs=-1).fit(
            np.vstack(histogram_features), labels_array
        )
        signature = dataset_signature(context.synthetic, columns, context.metadata.categorical_columns)
        target = np.vstack([context.real_train, context.real_holdout])
        candidate_features = encoder.transform(target)
        target_labels = np.concatenate([np.ones(len(context.real_train)), np.zeros(len(context.real_holdout))])
        naive_prob = naive.predict_proba(
            np.hstack([candidate_features, np.tile(signature[::2], (len(target), 1))])
        )[:, 1]
        hist_prob = histogram.predict_proba(
            np.hstack([candidate_features, np.tile(signature, (len(target), 1))])
        )[:, 1]
        naive_auc = safe_auc(naive_prob[target_labels == 1], naive_prob[target_labels == 0])
        hist_auc = safe_auc(hist_prob[target_labels == 1], hist_prob[target_labels == 0])
        return self.completed(
            {
                "f_naive_auc": naive_auc,
                "f_hist_auc": hist_auc,
                "headline_auc": max(naive_auc, hist_auc),
                "shadow_examples": int(len(labels_array)),
            },
            details={"feature_extractors": "candidate encoding plus summary or histogram release signatures"},
        )

