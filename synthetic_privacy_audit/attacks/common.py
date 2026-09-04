from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.features.tabular import TabularEncoder, nearest_indices


def is_continuous(frame: pd.DataFrame, column: str) -> bool:
    return pd.api.types.is_numeric_dtype(frame[column]) and frame[column].nunique(dropna=True) > 15


@dataclass
class FittedSensitiveModels:
    encoder: TabularEncoder
    models: dict[str, tuple[object, LabelEncoder | None, bool]]


def fit_sensitive_models(
    synthetic: pd.DataFrame,
    known_columns: Sequence[str],
    sensitive_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    seed: int,
) -> FittedSensitiveModels:
    encoder = TabularEncoder.fit(synthetic, known_columns, categorical_columns)
    features = encoder.transform(synthetic)
    models: dict[str, tuple[object, LabelEncoder | None, bool]] = {}
    for column in sensitive_columns:
        continuous = is_continuous(synthetic, column)
        if continuous:
            target = pd.to_numeric(synthetic[column], errors="coerce").fillna(
                pd.to_numeric(synthetic[column], errors="coerce").median()
            )
            model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
            model.fit(features, target)
            models[column] = (model, None, True)
        else:
            labels = LabelEncoder()
            target = labels.fit_transform(synthetic[column].astype("string").fillna("<missing>"))
            model = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(features, target)
            models[column] = (model, labels, False)
    return FittedSensitiveModels(encoder, models)


def predict_sensitive(models: FittedSensitiveModels, frame: pd.DataFrame) -> pd.DataFrame:
    features = models.encoder.transform(frame)
    values: dict[str, np.ndarray] = {}
    for column, (model, labels, continuous) in models.models.items():
        prediction = model.predict(features)
        values[column] = prediction if continuous else labels.inverse_transform(prediction)
    return pd.DataFrame(values, index=frame.index)


def sensitive_success(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    metadata: AttackContext,
) -> tuple[np.ndarray, dict[str, float]]:
    successes: list[np.ndarray] = []
    per_column: dict[str, float] = {}
    for column in prediction.columns:
        if is_continuous(truth, column):
            actual = pd.to_numeric(truth[column], errors="coerce").to_numpy(dtype=float)
            predicted = pd.to_numeric(prediction[column], errors="coerce").to_numpy(dtype=float)
            tolerance = metadata.metadata.continuous_tolerances.get(
                column,
                max(float(pd.to_numeric(truth[column], errors="coerce").std(ddof=0)) * 0.1, 1e-9),
            )
            correct = np.abs(actual - predicted) <= tolerance
        else:
            correct = (
                truth[column].astype("string").fillna("<missing>").to_numpy()
                == prediction[column].astype("string").fillna("<missing>").to_numpy()
            )
        successes.append(correct)
        per_column[f"{column}_success"] = float(np.mean(correct))
    return np.logical_and.reduce(successes), per_column


def nearest_sensitive_predictions(
    context: AttackContext,
    targets: pd.DataFrame,
    *,
    known_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    known = tuple(known_columns or context.metadata.known_columns)
    encoder = TabularEncoder.fit(
        context.real_train,
        known,
        context.metadata.categorical_columns,
    )
    target_array = encoder.transform(targets)
    synthetic_array = encoder.transform(context.synthetic)
    indices, distances = nearest_indices(target_array, synthetic_array)
    return context.synthetic.iloc[indices].loc[:, list(context.metadata.sensitive_columns)].reset_index(drop=True), distances


def membership_distance_scores(context: AttackContext, targets: pd.DataFrame, *, metric: str) -> np.ndarray:
    common = tuple(column for column in context.real_train.columns if column in context.synthetic.columns)
    encoder = TabularEncoder.fit(context.real_train, common, context.metadata.categorical_columns)
    indices, distances = nearest_indices(
        encoder.transform(targets),
        encoder.transform(context.synthetic),
        metric=metric,
    )
    del indices
    return -distances


def safe_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    if len(np.unique(labels)) < 2 or np.allclose(scores, scores[0]):
        return 0.5
    return float(roc_auc_score(labels, scores))


def top_quantile_indices(values: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    if not len(values):
        return np.asarray([], dtype=int)
    count = max(1, int(np.ceil(len(values) * fraction)))
    return np.argsort(values)[-count:]

