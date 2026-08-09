"""
Tests unitaires du pipeline ML (src/common.py).
Couvre les fonctions partagées utilisées par les scripts 05/08/09/12.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from common import (
    PERFORMANCE_THRESHOLD,
    build_preprocessor,
    compute_metrics,
    make_promotion_gate,
    remove_columns,
    select_column_types,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4],
            "date_demande": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "nom_partenaire": ["A", "B", "A", "C"],
            "age": [30, 45, 41, 22],
            "revenu_mensuel_xaf": [1000.0, 2000.0, 1500.0, None],
            "pays": ["CMR", "CMR", "COG", "CMR"],
        }
    )


def test_remove_columns_drops_noise_columns(sample_df):
    cleaned = remove_columns(sample_df)
    assert "applicant_id" not in cleaned.columns
    assert "date_demande" not in cleaned.columns
    assert "nom_partenaire" not in cleaned.columns
    assert "age" in cleaned.columns


def test_remove_columns_ignores_missing_columns():
    df = pd.DataFrame({"age": [1, 2]})
    assert list(remove_columns(df).columns) == ["age"]


def test_select_column_types(sample_df):
    sample_df = remove_columns(sample_df)
    numeric_cols, categorical_cols = select_column_types(sample_df)
    assert "age" in numeric_cols
    assert "pays" in categorical_cols


def test_build_preprocessor_pipeline(sample_df):
    sample_df = remove_columns(sample_df)
    numeric_cols, categorical_cols = select_column_types(sample_df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    assert hasattr(preprocessor, "fit_transform")
    assert isinstance(Pipeline([("preprocessor", preprocessor)]), Pipeline)


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_proba)
    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_compute_metrics_include_accuracy():
    y_true = np.array([0, 1])
    y_proba = np.array([0.1, 0.9])
    metrics = compute_metrics(y_true, y_proba, include_accuracy=True)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_make_promotion_gate_above_threshold():
    gate = make_promotion_gate(PERFORMANCE_THRESHOLD + 0.01)
    assert gate["passed"] is True
    assert gate["decision"] == "PROMOTE_TO_PRODUCTION"


def test_make_promotion_gate_below_threshold():
    gate = make_promotion_gate(PERFORMANCE_THRESHOLD - 0.01)
    assert gate["passed"] is False
    assert gate["decision"] == "NOT_PROMOTE"
