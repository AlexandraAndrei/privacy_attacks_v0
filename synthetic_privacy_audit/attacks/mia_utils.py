from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.statistics import dataset_signature, outlier_scores
from synthetic_privacy_audit.features.tabular import TabularEncoder, nearest_indices


def common_columns(context: AttackContext) -> tuple[str, ...]:
    columns = tuple(column for column in context.real_train.columns if column in context.synthetic.columns)
    if not columns:
        raise ValueError("No common columns between real and synthetic data.")
    return columns


def reference_population(context: AttackContext) -> pd.DataFrame:
    return context.reference_data if context.reference_data is not None else (
        context.real_control if context.real_control is not None else context.real_holdout
    )


def distance_scores_to_release(
    fit_frame: pd.DataFrame,
    release: pd.DataFrame,
    targets: pd.DataFrame,
    columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    metric: str = "euclidean",
) -> np.ndarray:
    encoder = TabularEncoder.fit(fit_frame, columns, categorical_columns)
    _, distances = nearest_indices(
        encoder.transform(targets),
        encoder.transform(release),
        metric=metric,
    )
    return -distances


def shadow_membership_signals(
    context: AttackContext,
    *,
    shadow_runs: int,
    candidates_per_run: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produces black-box shadow labels, nearest-release scores and rarity values."""
    if context.reference_data is None:
        raise ValueError("Requires reference_data for shadow membership examples.")
    columns = common_columns(context)
    reference = context.reference_data.reset_index(drop=True)
    scores, labels, difficulties = [], [], []
    for run in range(shadow_runs):
        members = reference.sample(frac=0.5, random_state=context.seed + run)
        release = context.generator.generate(members, seed=context.seed + run)
        candidates = reference.sample(
            n=min(candidates_per_run, len(reference)),
            random_state=context.seed + 100 + run,
        )
        in_shadow = candidates.index.isin(members.index).astype(int)
        scores.extend(
            distance_scores_to_release(
                reference, release, candidates, columns, context.metadata.categorical_columns, metric
            )
        )
        labels.extend(in_shadow)
        difficulties.extend(
            outlier_scores(
                reference,
                candidates,
                context.metadata.quasi_identifier_columns or columns,
                context.metadata.categorical_columns,
            )
        )
    return np.asarray(scores), np.asarray(difficulties), np.asarray(labels)


def target_signals(context: AttackContext, *, metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = common_columns(context)
    reference = reference_population(context)
    members = distance_scores_to_release(
        reference, context.synthetic, context.real_train, columns, context.metadata.categorical_columns, metric
    )
    nonmembers = distance_scores_to_release(
        reference, context.synthetic, context.real_holdout, columns, context.metadata.categorical_columns, metric
    )
    difficulties = np.concatenate(
        [
            outlier_scores(reference, context.real_train, context.metadata.quasi_identifier_columns or columns, context.metadata.categorical_columns),
            outlier_scores(reference, context.real_holdout, context.metadata.quasi_identifier_columns or columns, context.metadata.categorical_columns),
        ]
    )
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    return np.concatenate([members, nonmembers]), difficulties, labels


def release_signature(context: AttackContext, release: pd.DataFrame) -> np.ndarray:
    columns = common_columns(context)
    return dataset_signature(release, columns, context.metadata.categorical_columns)

