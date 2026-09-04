from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder, nearest_indices
from synthetic_privacy_audit.types import AttackResult


class TemporalLinkabilityAttack(BaseAttack):
    """Temporal nearest-neighbour linkage proxy across time-sliced synthetic releases."""

    name = "temporal_linkability"

    def run(self, context: AttackContext) -> AttackResult:
        self.require_releases(context, 2)
        temporal = context.metadata.temporal_column
        if not temporal:
            raise AttackPrerequisiteError("Requires temporal_column in dataset configuration.")
        if temporal not in context.synthetic.columns or temporal not in context.synthetic_releases[0].columns:
            raise ValueError(f"Temporal column {temporal} is absent from at least one release.")
        columns = tuple(column for column in context.metadata.known_columns if column != temporal)
        if not columns:
            raise ValueError("Temporal linkage needs a non-temporal known attribute.")
        encoder = TabularEncoder.fit(context.real_train, columns, context.metadata.categorical_columns)
        left, right = context.synthetic, context.synthetic_releases[0]
        indices, distances = nearest_indices(encoder.transform(left), encoder.transform(right))
        left_time = pd.to_numeric(left[temporal], errors="coerce").to_numpy(dtype=float)
        right_time = pd.to_numeric(right.iloc[indices][temporal], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(left_time) & np.isfinite(right_time)
        if not finite.any():
            raise ValueError("Temporal column must be numeric or pre-converted before running this attack.")
        gaps = np.abs(left_time[finite] - right_time[finite])
        return self.completed(
            {
                "evaluated_links": int(finite.sum()),
                "median_temporal_gap": float(np.median(gaps)),
                "mean_link_distance": float(distances[finite].mean()),
            },
            details={"proxy": "nearest non-temporal match across first two releases, followed by temporal-gap analysis"},
        )

