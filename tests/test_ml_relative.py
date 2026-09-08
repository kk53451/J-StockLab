import numpy as np
import pandas as pd
import pytest

from ml_eval.data import STOCKS, DataConfig, make_dataset
from ml_eval.run import predict


def dataset_inputs():
    dates = pd.bdate_range("2020-01-01", periods=80)
    rng = np.random.default_rng(27)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (80, len(STOCKS))), axis=0))
    frame = pd.DataFrame(values, index=dates, columns=STOCKS)
    config = DataConfig(5, 3, str(dates[39].date()), str(dates[59].date()), representation="relative")
    return frame, config


def test_relative_inputs_end_at_zero_and_targets_decode_to_prices():
    frame, config = dataset_inputs()
    result = make_dataset(frame, list(STOCKS), config)
    assert result.scalers["x"] is None
    for windows in result.partitions.values():
        np.testing.assert_array_equal(windows.x[:, -1], 0)
        returns = result.scalers["y"].inverse_transform(windows.y_scaled.reshape(-1, len(STOCKS))).reshape(windows.y.shape)
        np.testing.assert_allclose(windows.current[:, None] * (1 + returns), windows.y, atol=2e-5)
        origin_idx = frame.index.get_loc(windows.origins[0])
        expected = frame.iloc[origin_idx - 4:origin_idx + 1].to_numpy() / windows.current[0] - 1
        np.testing.assert_allclose(windows.x[0], expected, atol=1e-7)


def test_relative_representation_is_invariant_to_stock_price_units():
    frame, config = dataset_inputs()
    original = make_dataset(frame, list(STOCKS), config)
    scaled = make_dataset(frame * np.arange(1, len(STOCKS) + 1), list(STOCKS), config)
    for name in original.partitions:
        np.testing.assert_allclose(original.partitions[name].x, scaled.partitions[name].x, atol=1e-7)
        np.testing.assert_allclose(original.partitions[name].y_scaled, scaled.partitions[name].y_scaled, atol=1e-6)


def test_future_values_cannot_affect_relative_training_or_target_scaler():
    frame, config = dataset_inputs()
    original = make_dataset(frame, list(STOCKS), config)
    modified = frame.copy()
    modified.loc[modified.index > config.train_end] *= 10000
    future = make_dataset(modified, list(STOCKS), config)
    np.testing.assert_array_equal(original.scalers["y"].data_max_, future.scalers["y"].data_max_)
    np.testing.assert_array_equal(original.partitions["train"].x, future.partitions["train"].x)
    np.testing.assert_array_equal(original.partitions["train"].y_scaled, future.partitions["train"].y_scaled)


def test_zero_return_prediction_recovers_persistence():
    frame, config = dataset_inputs()
    result = make_dataset(frame, list(STOCKS), config)
    windows = result.partitions["validation"]

    class ZeroReturnModel:
        def predict(self, x):
            return result.scalers["y"].transform(np.zeros((len(x) * 3, len(STOCKS)))).reshape(len(x), -1)

    prices = predict("ridge", ZeroReturnModel(), windows, result.scalers, 32)
    np.testing.assert_allclose(prices, np.broadcast_to(windows.current[:, None], windows.y.shape), atol=1e-10)


def test_mismatched_saved_representation_is_rejected():
    frame, config = dataset_inputs()
    result = make_dataset(frame, list(STOCKS), config)
    price_config = DataConfig(5, 3, config.train_end, config.validation_end)
    with pytest.raises(ValueError, match="representation"):
        make_dataset(frame, list(STOCKS), price_config, result.scalers)


def test_economic_features_are_not_silently_origin_normalized():
    with pytest.raises(ValueError, match="stock-only"):
        DataConfig(feature_set="stock-econ", representation="relative").validate()


def test_price_artifacts_without_representation_remain_compatible():
    frame, config = dataset_inputs()
    price_config = DataConfig(5, 3, config.train_end, config.validation_end)
    original = make_dataset(frame, list(STOCKS), price_config)
    legacy_scalers = {k: v for k, v in original.scalers.items()
                      if k not in ("representation", "input_representation", "target_representation")}
    restored = make_dataset(frame, list(STOCKS), price_config, legacy_scalers)
    windows = restored.partitions["validation"]

    class EchoTargets:
        def predict(self, x):
            return windows.y_scaled.reshape(len(x), -1)

    np.testing.assert_array_equal(windows.x, original.partitions["validation"].x)
    np.testing.assert_allclose(predict("ridge", EchoTargets(), windows, legacy_scalers, 32), windows.y, atol=2e-5)
