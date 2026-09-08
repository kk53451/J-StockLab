import numpy as np
import pandas as pd
import pytest

from ml_eval.data import STOCKS, DataConfig, load_frame, make_dataset


@pytest.fixture
def frame():
    dates = pd.bdate_range("2020-01-01", periods=80)
    values = np.arange(80)[:, None] + np.arange(len(STOCKS))[None, :] + 100.0
    return pd.DataFrame(values, index=dates, columns=STOCKS)


def config(frame):
    return DataConfig(lookback=5, horizon=3, train_end=str(frame.index[39].date()),
                      validation_end=str(frame.index[59].date()))


def test_future_extremes_cannot_change_scalers_or_training(frame):
    settings = config(frame)
    original = make_dataset(frame, list(STOCKS), settings)
    changed = frame.copy()
    changed.loc[changed.index > settings.train_end] *= 1000
    future = make_dataset(changed, list(STOCKS), settings)
    np.testing.assert_array_equal(original.scalers["x"].data_max_, future.scalers["x"].data_max_)
    np.testing.assert_array_equal(original.scalers["y"].data_max_, future.scalers["y"].data_max_)
    np.testing.assert_array_equal(original.partitions["train"].x, future.partitions["train"].x)
    np.testing.assert_array_equal(original.partitions["train"].y_scaled, future.partitions["train"].y_scaled)
    assert future.partitions["test"].x.max() > 1  # No clipping to hide distribution shift.


def test_all_target_dates_are_disjoint_across_partitions(frame):
    result = make_dataset(frame, list(STOCKS), config(frame))
    target_sets = [set(w.target_dates.flatten()) for w in result.partitions.values()]
    assert target_sets[0].isdisjoint(target_sets[1])
    assert target_sets[0].isdisjoint(target_sets[2])
    assert target_sets[1].isdisjoint(target_sets[2])
    assert result.metadata["purged_boundary_windows"] == 4


def test_origin_is_last_input_and_day1_is_next_observation(frame):
    result = make_dataset(frame, list(STOCKS), config(frame))
    for windows in result.partitions.values():
        last_input = result.scalers["x"].inverse_transform(windows.x[:, -1])
        np.testing.assert_allclose(last_input, windows.current, atol=1e-4)
        for i, origin in enumerate(windows.origins):
            position = frame.index.get_loc(origin)
            assert windows.target_dates[i, 0] == frame.index[position + 1]
            np.testing.assert_array_equal(windows.y[i], frame.iloc[position + 1:position + 4].to_numpy())


def test_saved_scalers_are_used_without_refitting(frame):
    settings = config(frame)
    result = make_dataset(frame, list(STOCKS), settings)
    changed = frame * 2
    restored = make_dataset(changed, list(STOCKS), settings, result.scalers)
    assert restored.scalers is result.scalers
    np.testing.assert_array_equal(restored.scalers["x"].data_max_, frame.iloc[:40].max().to_numpy())


def calendar_csv(tmp_path):
    dates = pd.date_range("2024-01-01", "2024-01-10")
    frame = pd.DataFrame({s: np.arange(len(dates)) + 100.0 for s in STOCKS})
    frame.insert(0, "날짜", dates)
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_japanese_holidays_and_weekends_are_not_horizons(tmp_path):
    path, _ = calendar_csv(tmp_path)
    frame, _, info = load_frame(path, "stock")
    assert frame.index.strftime("%Y-%m-%d").tolist() == ["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10"]
    assert info["removed_non_sessions"] == 6


@pytest.mark.parametrize("issue", ["missing_session", "nan_price", "duplicate_date"])
def test_invalid_observations_are_rejected_not_filled(tmp_path, issue):
    path, data = calendar_csv(tmp_path)
    if issue == "missing_session":
        data = data.drop(4)
    elif issue == "nan_price":
        data.loc[4, "Toyota"] = np.nan
    else:
        data = pd.concat([data, data.iloc[[4]]])
    data.to_csv(path, index=False)
    expected = {"missing_session": "Missing trading sessions", "nan_price": "Non-finite", "duplicate_date": "unique"}
    with pytest.raises(ValueError, match=expected[issue]):
        load_frame(path, "stock")


def test_economic_features_are_lagged_by_session_and_flagged(tmp_path):
    path, data = calendar_csv(tmp_path)
    data["economic"] = np.arange(len(data)) * 10.0
    data.to_csv(path, index=False)
    frame, features, info = load_frame(path, "stock-econ")
    assert frame.loc["2024-01-05", "economic"] == 30.0  # January 4, not same-date value.
    assert frame.loc["2024-01-09", "economic"] == 40.0  # Previous trading session January 5.
    assert features[-1] == "economic"
    assert any("EXPLORATORY" in item for item in info["limitations"])


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"lookback": -1}, {"train_end": "2025-01-01"}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs).validate()
