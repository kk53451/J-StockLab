import numpy as np
import pandas as pd
import pytest

from ml_eval.data import STOCKS, DataConfig, make_dataset
from ml_eval.run import predict


def inputs(input_mode, target_mode):
    dates = pd.bdate_range("2020-01-01", periods=80)
    rng = np.random.default_rng(4)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (80, len(STOCKS))), axis=0))
    frame = pd.DataFrame(values, index=dates, columns=STOCKS)
    config = DataConfig(5, 3, str(dates[39].date()), str(dates[59].date()),
                        input_representation=input_mode, target_representation=target_mode)
    return frame, config


@pytest.mark.parametrize("input_mode", ["price", "relative"])
@pytest.mark.parametrize("target_mode", ["price", "relative"])
def test_independent_modes_preserve_input_and_price_decoding(input_mode, target_mode):
    frame, config = inputs(input_mode, target_mode)
    dataset = make_dataset(frame, list(STOCKS), config)
    windows = dataset.partitions["validation"]
    if input_mode == "relative":
        np.testing.assert_array_equal(windows.x[:, -1], 0)
    else:
        np.testing.assert_allclose(dataset.scalers["x"].inverse_transform(windows.x[:, -1]), windows.current, atol=2e-5)

    class PerfectModel:
        def predict(self, x):
            return windows.y_scaled.reshape(len(x), -1)

    np.testing.assert_allclose(predict("ridge", PerfectModel(), windows, dataset.scalers, 32), windows.y, atol=2e-5)
    perturbed = frame.copy()
    perturbed.loc[perturbed.index > config.train_end] *= 10000
    future = make_dataset(perturbed, list(STOCKS), config)
    np.testing.assert_array_equal(future.scalers["y"].data_max_, dataset.scalers["y"].data_max_)
    np.testing.assert_array_equal(future.partitions["train"].x, dataset.partitions["train"].x)
    np.testing.assert_array_equal(future.partitions["train"].y_scaled, dataset.partitions["train"].y_scaled)


def test_preset_compatibility_and_override_resolution():
    assert DataConfig(representation="relative").input_mode == "relative"
    assert DataConfig(representation="relative").target_mode == "relative"
    config = DataConfig(representation="relative", input_representation="price")
    assert config.input_mode == "price" and config.target_mode == "relative"


def test_previous_relative_artifact_without_independent_modes_loads():
    frame, config = inputs("relative", "relative")
    dataset = make_dataset(frame, list(STOCKS), config)
    legacy = {k: v for k, v in dataset.scalers.items() if k not in ("input_representation", "target_representation")}
    legacy["representation"] = "relative"
    restored = make_dataset(frame, list(STOCKS), config, legacy)
    np.testing.assert_array_equal(restored.partitions["validation"].x, dataset.partitions["validation"].x)


def test_only_target_mismatch_is_rejected():
    frame, config = inputs("price", "relative")
    dataset = make_dataset(frame, list(STOCKS), config)
    _, wrong_config = inputs("price", "price")
    with pytest.raises(ValueError, match="representation"):
        make_dataset(frame, list(STOCKS), wrong_config, dataset.scalers)
