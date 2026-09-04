from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


def _predicate_unique_rate(
    synthetic: pd.DataFrame,
    target: pd.DataFrame,
    columns: tuple[str, ...],
    width: int,
    sample_size: int,
    seed: int,
) -> float:
    sampled = synthetic.sample(n=min(sample_size, len(synthetic)), random_state=seed)
    outcomes = []
    for _, row in sampled.iterrows():
        selected = columns[:width]
        mask = np.ones(len(target), dtype=bool)
        for column in selected:
            mask &= target[column].astype("string").to_numpy() == str(row[column])
        outcomes.append(mask.sum() == 1)
    return float(np.mean(outcomes)) if outcomes else 0.0


class SinglingOutAttack(BaseAttack):
    """Anonymeter-style univariate and multivariate predicate singling-out."""

    name = "singling_out"

    def __init__(self, sample_size: int = 1_000) -> None:
        self.sample_size = sample_size

    def run(self, context: AttackContext) -> AttackResult:
        if context.real_control is None:
            raise AttackPrerequisiteError("Requires real_control.csv to estimate baseline singling-out risk.")
        qis = context.metadata.quasi_identifier_columns
        self.require_columns(context, qis)
        if not qis:
            raise ValueError("Singling-out requires quasi_identifier_columns.")
        univariate_train = _predicate_unique_rate(
            context.synthetic, context.real_train, qis, 1, self.sample_size, context.seed
        )
        univariate_control = _predicate_unique_rate(
            context.synthetic, context.real_control, qis, 1, self.sample_size, context.seed
        )
        width = min(3, len(qis))
        multivariate_train = _predicate_unique_rate(
            context.synthetic, context.real_train, qis, width, self.sample_size, context.seed + 1
        )
        multivariate_control = _predicate_unique_rate(
            context.synthetic, context.real_control, qis, width, self.sample_size, context.seed + 1
        )
        return self.completed(
            {
                "univariate_train_unique_rate": univariate_train,
                "univariate_excess_over_control": univariate_train - univariate_control,
                "multivariate_train_unique_rate": multivariate_train,
                "multivariate_excess_over_control": multivariate_train - multivariate_control,
            },
            details={"predicate_width": width, "success": "predicate isolates exactly one real row"},
        )

