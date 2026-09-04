from __future__ import annotations

import pandas as pd

from synthetic_privacy_audit.attacks.base import BaseAttack
from synthetic_privacy_audit.attacks.common import sensitive_success
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class RAPReconstructionAttack(BaseAttack):
    """k-way marginal reconstruction proxy based on synthetic conditional tables."""

    name = "rap_joint_reconstruction"

    def run(self, context: AttackContext) -> AttackResult:
        known = tuple(context.metadata.known_columns[: min(3, len(context.metadata.known_columns))])
        sensitive = context.metadata.sensitive_columns
        self.require_columns(context, [*known, *sensitive])
        if not known:
            raise ValueError("RAP reconstruction needs at least one known attribute.")
        synthetic = context.synthetic.copy()
        target = context.real_holdout.copy()
        synthetic["_known_key"] = synthetic.loc[:, list(known)].astype("string").agg("|".join, axis=1)
        target["_known_key"] = target.loc[:, list(known)].astype("string").agg("|".join, axis=1)
        synthetic["_secret_key"] = synthetic.loc[:, list(sensitive)].astype("string").agg("|".join, axis=1)
        modal = (
            synthetic.groupby("_known_key")["_secret_key"]
            .agg(lambda series: series.value_counts().index[0])
            .to_dict()
        )
        fallback = synthetic["_secret_key"].value_counts().index[0]
        reconstructed = target["_known_key"].map(modal).fillna(fallback).str.split("|", expand=True)
        reconstructed.columns = list(sensitive)
        success, per_column = sensitive_success(target.reset_index(drop=True), reconstructed, context)
        return self.completed(
            {
                "joint_reconstruction_success": float(success.mean()),
                "known_attribute_arity": int(len(known)),
                "unseen_known_pattern_rate": float((~target["_known_key"].isin(modal)).mean()),
                **per_column,
            },
            details={"mechanism": "modal sensitive tuple from synthetic k-way conditionals"},
        )

