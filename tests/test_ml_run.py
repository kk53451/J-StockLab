import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_eval.data import STOCKS, DataConfig, load_frame, make_dataset
from ml_eval.run import evaluate_test, parser, train


@pytest.fixture(params=[
    ("ridge", "price", "price"), ("ridge", "price", "relative"),
    ("ridge", "relative", "price"), ("ridge", "relative", "relative"),
    ("mean-return", "price", "relative"),
])
def run(tmp_path, request):
    dates = pd.date_range("2020-01-01", "2020-06-30")
    rng = np.random.default_rng(12)
    prices = 100 + np.cumsum(rng.normal(0.1, 0.3, (len(dates), len(STOCKS))), axis=0)
    data = pd.DataFrame(prices, columns=STOCKS)
    data.insert(0, "날짜", dates)
    source = tmp_path / "data.csv"
    data.to_csv(source, index=False)
    output = tmp_path / "run"
    args = parser().parse_args([
        "train", "--data", str(source), "--output", str(output), "--model", request.param[0],
        "--lookback", "5", "--horizon", "3", "--train-end", "2020-02-28",
        "--validation-end", "2020-03-31",
        "--input-representation", request.param[1], "--target-representation", request.param[2],
    ])
    train(args)
    return output, source


def test_train_does_not_evaluate_test_and_frozen_model_can_be_evaluated(run):
    output, source = run
    assert (output / "validation" / "summary.json").exists()
    assert not (output / "test").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    if manifest["model"] == "mean-return":
        frame, features, _ = load_frame(source, "stock")
        training = make_dataset(frame, features, DataConfig(**manifest["data_config"])).partitions["train"]
        expected = (training.y / training.current[:, None] - 1).mean(axis=0)
        np.testing.assert_allclose(joblib.load(output / "model.joblib"), expected)
    args = parser().parse_args(["evaluate-test", "--run", str(output)])
    evaluate_test(args)
    predictions = pd.read_csv(output / "test" / "predictions.csv")
    assert (predictions.target_date > "2020-03-31").all()
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_test(args)


def test_changed_data_cannot_be_used_with_saved_model(run):
    output, source = run
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Data hash mismatch"):
        evaluate_test(parser().parse_args(["evaluate-test", "--run", str(output)]))
    assert not (output / "test").exists()


def test_saved_model_tampering_is_detected(run):
    output, _ = run
    with (output / "model.joblib").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        evaluate_test(parser().parse_args(["evaluate-test", "--run", str(output)]))
