from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class StatisticalPropertyInferenceAttack(BaseAttack):
    """Infers training-population proportions/statistics directly from a release."""

    name = "statistical_property_inference"

    def run(self, context: AttackContext) -> AttackResult:
        columns = context.metadata.property_columns
        self.require_columns(context, columns)
        divergences = {}
        for column in columns:
            if pd.api.types.is_numeric_dtype(context.real_train[column]):
                train = pd.to_numeric(context.real_train[column], errors="coerce")
                synthetic = pd.to_numeric(context.synthetic[column], errors="coerce")
                scale = max(float(train.std(ddof=0)), 1e-9)
                divergences[f"{column}_standardized_mean_error"] = float(abs(train.mean() - synthetic.mean()) / scale)
            else:
                train = context.real_train[column].astype("string").value_counts(normalize=True)
                synthetic = context.synthetic[column].astype("string").value_counts(normalize=True)
                labels = train.index.union(synthetic.index)
                divergences[f"{column}_total_variation"] = float(
                    0.5 * np.abs(train.reindex(labels, fill_value=0.0) - synthetic.reindex(labels, fill_value=0.0)).sum()
                )
        return self.completed(
            {
                "mean_property_divergence": float(np.mean(list(divergences.values()))),
                **divergences,
            },
            details={"interpretation": "low divergence means release exposes a more accurate group-level property"},
        )

