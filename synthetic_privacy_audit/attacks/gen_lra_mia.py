from __future__ import annotations

from sklearn.neighbors import KernelDensity

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import safe_auc
from synthetic_privacy_audit.attacks.mia_utils import common_columns, reference_population
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder
from synthetic_privacy_audit.types import AttackResult


class GenLikelihoodRatioMembershipInferenceAttack(BaseAttack):
    """Gen-LRA: no-box KDE likelihood ratio between release and population."""

    name = "gen_lra_membership_inference"

    def run(self, context: AttackContext) -> AttackResult:
        reference = reference_population(context)
        columns = common_columns(context)
        encoder = TabularEncoder.fit(reference, columns, context.metadata.categorical_columns)
        synthetic_kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(encoder.transform(context.synthetic))
        population_kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(encoder.transform(reference))

        def ratio(frame):
            values = encoder.transform(frame)
            return synthetic_kde.score_samples(values) - population_kde.score_samples(values)

        member_ratio, nonmember_ratio = ratio(context.real_train), ratio(context.real_holdout)
        return self.completed(
            {
                "gen_lra_auc": safe_auc(member_ratio, nonmember_ratio),
                "member_mean_log_likelihood_ratio": float(member_ratio.mean()),
                "nonmember_mean_log_likelihood_ratio": float(nonmember_ratio.mean()),
            },
            details={"test": "KDE synthetic density divided by KDE reference-population density"},
        )

