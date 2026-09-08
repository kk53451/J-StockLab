import json

import numpy as np
import pytest

from ml_eval.data import STOCKS, Windows
from ml_eval.metrics import persistence, prediction_table, score, yearly_scores


def windows(flat=False):
    shape = (2, 2, len(STOCKS))
    actual = np.full(shape, 100.0 if flat else 110.0)
    return Windows(np.zeros((2, 3, len(STOCKS))), actual, actual,
                   np.full((2, len(STOCKS)), 100.0),
                   np.array(["2024-01-04", "2024-01-05"], dtype="datetime64[D]"),
                   np.array([["2024-01-05", "2024-01-09"], ["2024-01-09", "2024-01-10"]], dtype="datetime64[D]"))


def test_perfect_forecast_and_persistence_have_expected_skill():
    data = windows()
    _, perfect = score(data, data.y)
    table, baseline = score(data, persistence(data))
    assert perfect["macro_mape_pct"] == 0
    assert perfect["macro_mae_skill_vs_persistence"] == 1
    assert perfect["macro_direction_agreement_pct"] == 100
    assert baseline["macro_mae_skill_vs_persistence"] == 0
    assert baseline["macro_direction_agreement_pct"] == 0
    assert table.iloc[0]["mae"] == 10
    assert table.iloc[0]["mape_pct"] == pytest.approx(100 / 11)


def test_zero_baseline_error_is_null_not_infinity():
    data = windows(flat=True)
    _, summary = score(data, persistence(data))
    assert summary["macro_mae_skill_vs_persistence"] is None
    assert summary["macro_direction_agreement_pct"] == 100
    json.dumps(summary, allow_nan=False)


def test_prediction_export_preserves_origin_target_stock_order():
    data = windows()
    table = prediction_table(data, data.y)
    assert len(table) == 80
    assert table.iloc[0]["stock"] == STOCKS[0]
    assert table.iloc[20]["horizon_sessions"] == 2
    assert str(table.iloc[20]["target_date"].date()) == "2024-01-09"
    assert str(table.iloc[40]["origin_date"].date()) == "2024-01-05"
    assert table.iloc[0]["predicted_return_pct"] == pytest.approx(10)


def test_nonfinite_predictions_fail():
    data = windows()
    predictions = data.y.copy()
    predictions[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        score(data, predictions)


def test_year_slices_exclude_only_cross_year_target_windows():
    data = windows()
    data.target_dates = np.array([["2023-12-29", "2024-01-04"], ["2024-01-04", "2024-01-05"]], dtype="datetime64[D]")
    result = yearly_scores(data, data.y)
    assert result["excluded_cross_year_windows"] == 1
    assert list(result["years"]) == ["2024"]
    assert result["years"]["2024"]["model"]["origin_count"] == 1
    assert result["years"]["2024"]["model"]["macro_mape_pct"] == 0
    assert result["years"]["2024"]["persistence"]["macro_mape_pct"] > 0
    assert score(data, data.y)[1]["origin_count"] == 2


def test_year_slices_reject_wrong_shape():
    data = windows()
    with pytest.raises(ValueError, match="shape"):
        yearly_scores(data, data.y[:1])
