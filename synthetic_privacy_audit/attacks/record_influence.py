from __future__ import annotations

import numpy as np

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import top_quantile_indices
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import outlier_scores
from synthetic_privacy_audit.features.tabular import TabularEncoder, nearest_indices
from synthetic_privacy_audit.types import AttackResult


class RecordInfluenceAttack(BaseAttack):
    """Leave-one-out influence of high-rarity training records on a release."""

    name = "record_influence"

    def __init__(self, target_count: int = 10) -> None:
        self.target_count = target_count

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        columns = tuple(column for column in context.real_train.columns if column in context.synthetic.columns)
        if not columns:
            raise ValueError("No common columns available for record-influence distance.")
        rarity = outlier_scores(
            context.real_train,
            context.real_train,
            context.metadata.quasi_identifier_columns or columns,
            context.metadata.categorical_columns,
        )
        selected = top_quantile_indices(rarity, min(1.0, self.target_count / len(context.real_train)))
        selected = selected[-self.target_count :]
        baseline = context.generator.generate(context.real_train, seed=context.seed)
        encoder = TabularEncoder.fit(context.real_train, columns, context.metadata.categorical_columns)
        influences = []
        for offset, row_index in enumerate(selected):
            target = context.real_train.iloc[[row_index]]
            _, baseline_distance = nearest_indices(encoder.transform(target), encoder.transform(baseline))
            without_target = context.real_train.drop(context.real_train.index[row_index])
            counterfactual = context.generator.generate(without_target, seed=context.seed + offset + 1)
            _, counterfactual_distance = nearest_indices(
                encoder.transform(target), encoder.transform(counterfactual)
            )
            influences.append(float(abs(counterfactual_distance[0] - baseline_distance[0])))
        return self.completed(
            {
                "evaluated_records": int(len(influences)),
                "mean_dcr_change": float(np.mean(influences)),
                "max_dcr_change": float(np.max(influences)),
            },
            details={"targets": "highest-rarity training rows", "measure": "leave-one-out closest-distance change"},
        )

