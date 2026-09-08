import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_eval.data import STOCKS, DataConfig, load_frame, make_dataset
from ml_eval.run import predict
from ml_eval.walk_forward import Fold, quarterly_folds, run_walk_forward, validate_folds


FOLDS = (Fold("first", "2020-02-28", "2020-04-30"), Fold("second", "2020-04-30", "2020-06-30"))


def create_csv(path):
    dates = pd.date_range("2020-01-01", "2020-09-30")
    rng = np.random.default_rng(14)
    values = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (len(dates), len(STOCKS))), axis=0))
    frame = pd.DataFrame(values, columns=STOCKS)
    frame.insert(0, "날짜", dates)
    frame.to_csv(path, index=False)
    return frame


def test_quarter_plan_is_contiguous_and_stays_out_of_reserved_period():
    folds = quarterly_folds()
    assert len(folds) == 8
    assert folds[0].train_end == "2021-12-31"
    assert folds[-1].evaluation_end == "2023-12-31"
    validate_folds(folds)


@pytest.mark.parametrize("folds", [
    (), (Fold("late", "2023-12-31", "2024-03-31"),),
    (Fold("a", "2020-01-31", "2020-03-31"), Fold("b", "2020-03-01", "2020-06-30")),
    (Fold("bad", "2020-03-31", "2020-01-31"),),
])
def test_invalid_fold_plan_is_rejected(folds):
    with pytest.raises(ValueError):
        validate_folds(folds)


def test_train_validation_only_never_materializes_later_windows(tmp_path):
    source = tmp_path / "data.csv"
    create_csv(source)
    frame, features, _ = load_frame(source, "stock")
    config = DataConfig(5, 3, FOLDS[0].train_end, FOLDS[0].evaluation_end, representation="relative")
    result = make_dataset(frame, features, config, include_test=False)
    truncated = make_dataset(frame.loc[:config.validation_end], features, config, include_test=False)
    assert set(result.partitions) == {"train", "validation"}
    for name, windows in result.partitions.items():
        assert windows.target_dates.max() <= np.datetime64(config.validation_end)
        np.testing.assert_array_equal(windows.x, truncated.partitions[name].x)
        np.testing.assert_array_equal(windows.y, truncated.partitions[name].y)


def test_loader_cuts_reserved_values_before_numeric_validation(tmp_path):
    source = tmp_path / "data.csv"
    raw = create_csv(source)
    future = raw.iloc[[0]].copy()
    future["날짜"] = pd.Timestamp("2024-01-04")
    future[list(STOCKS)] = np.nan
    pd.concat([raw, future]).to_csv(source, index=False)
    frame, _, metadata = load_frame(source, "stock", end_date="2020-06-30")
    assert frame.index.max() == pd.Timestamp("2020-06-30")
    assert metadata["excluded_after_end_date"] > 0
    assert not frame.isna().any().any()


def test_future_change_cannot_change_first_fold_and_fixed_control_is_frozen(tmp_path):
    source = tmp_path / "data.csv"
    raw = create_csv(source)
    output = tmp_path / "study"
    result = run_walk_forward(source, output, folds=FOLDS, lookback=5, horizon=3)
    changed = raw.copy()
    changed.loc[changed["날짜"] > pd.Timestamp(FOLDS[0].evaluation_end), list(STOCKS)] *= 1.7
    alternate = tmp_path / "changed.csv"
    changed.to_csv(alternate, index=False)
    changed_output = tmp_path / "changed-study"
    run_walk_forward(alternate, changed_output, folds=FOLDS, lookback=5, horizon=3)
    for name in result["models"]:
        before = pd.read_csv(output / "first" / name / "predictions.csv")
        after = pd.read_csv(changed_output / "first" / name / "predictions.csv")
        pd.testing.assert_frame_equal(before, after)
    # All expanding and frozen controls see identical evaluation targets.
    first = pd.read_csv(output / "first/ridge-expanding/predictions.csv")
    second = pd.read_csv(output / "second/ridge-expanding/predictions.csv")
    assert set(first.target_date).isdisjoint(set(second.target_date))
    pd.testing.assert_frame_equal(first, pd.read_csv(output / "first/ridge-fixed/predictions.csv"))
    assert result["folds"][1]["dataset"]["partitions"]["train"]["samples"] > result["folds"][0]["dataset"]["partitions"]["train"]["samples"]
    assert result["folds"][1]["fixed_control_fit_end"] == "2020-02-28"
    # Reproduce the second fold frozen control from ONLY first-fold artifacts.
    frame, features, _ = load_frame(source, "stock", end_date=FOLDS[1].evaluation_end)
    config = DataConfig(5, 3, FOLDS[1].train_end, FOLDS[1].evaluation_end, representation="relative")
    validation = make_dataset(frame, features, config, include_test=False).partitions["validation"]
    frozen = joblib.load(output / "first/ridge.joblib")
    scalers = joblib.load(output / "first/scalers.joblib")
    expected = predict("ridge", frozen, validation, scalers, 32)
    actual = pd.read_csv(output / "second/ridge-fixed/predictions.csv").predicted_price.to_numpy().reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete"
    assert not list(output.rglob("test"))
    pooled = pd.read_csv(output / "aggregate/ridge-expanding-predictions.csv")
    assert len(pooled) == len(first) + len(second)
    assert set(pooled.fold) == {"first", "second"}


def test_run_does_not_overwrite_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        run_walk_forward(tmp_path / "missing.csv", tmp_path, folds=FOLDS)
