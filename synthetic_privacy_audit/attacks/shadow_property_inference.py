from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import dataset_signature
from synthetic_privacy_audit.types import AttackResult


def _property_indicator(frame: pd.DataFrame, column: str) -> np.ndarray:
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce") >= pd.to_numeric(series, errors="coerce").median()).to_numpy()
    rare_label = series.astype("string").value_counts().index[-1]
    return (series.astype("string") == rare_label).to_numpy()


def _resample_property(frame: pd.DataFrame, column: str, high: bool, size: int, seed: int) -> pd.DataFrame:
    indicator = _property_indicator(frame, column)
    positive, negative = frame.loc[indicator], frame.loc[~indicator]
    if positive.empty or negative.empty:
        raise ValueError(f"Property column {column} has no usable two-sided split.")
    ratio = 0.8 if high else 0.2
    n_positive = int(round(size * ratio))
    return pd.concat(
        [
            positive.sample(n=n_positive, replace=len(positive) < n_positive, random_state=seed),
            negative.sample(n=size - n_positive, replace=len(negative) < size - n_positive, random_state=seed + 1),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed + 2)


class ShadowPropertyInferenceAttack(BaseAttack):
    """Black-box PIA using releases from shadow generators with two property buckets."""

    name = "shadow_property_inference_black_box"

    def __init__(self, shadow_runs: int = 10) -> None:
        self.shadow_runs = shadow_runs

    def run(self, context: AttackContext) -> AttackResult:
        self.require_generator(context)
        if context.reference_data is None:
            raise AttackPrerequisiteError("Requires reference_data for shadow property inference.")
        columns = context.metadata.property_columns
        self.require_columns(context, columns)
        property_column = columns[0]
        signatures, labels = [], []
        size = min(len(context.reference_data), len(context.real_train))
        for run in range(self.shadow_runs):
            high = bool(run % 2)
            training_set = _resample_property(
                context.reference_data, property_column, high, size, context.seed + run * 11
            )
            release = context.generator.generate(training_set, seed=context.seed + run)
            signatures.append(
                dataset_signature(release, columns, context.metadata.categorical_columns)
            )
            labels.append(int(high))
        classifier = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=context.seed, n_jobs=-1
        ).fit(np.vstack(signatures), labels)
        target_signature = dataset_signature(context.synthetic, columns, context.metadata.categorical_columns)
        probability_high = float(classifier.predict_proba(target_signature.reshape(1, -1))[0, 1])
        target_prevalence = float(_property_indicator(context.real_train, property_column).mean())
        return self.completed(
            {
                "property_high_probability": probability_high,
                "inferred_property_bucket": int(probability_high >= 0.5),
                "actual_train_property_prevalence": target_prevalence,
                "shadow_runs": self.shadow_runs,
            },
            details={
                "property_column": property_column,
                "threat_model": "black-box. White-box PIA needs a model-parameter adapter and is intentionally not simulated.",
            },
        )

