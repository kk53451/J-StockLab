"""CLI: fit/select on validation, then explicitly evaluate a frozen run on test."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .data import DataConfig, STOCKS, load_frame, make_dataset
from .metrics import persistence, prediction_table, score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def neural_module():
    try:
        from . import neural
        return neural
    except ModuleNotFoundError as error:
        if error.name == "tensorflow":
            raise RuntimeError("Install requirements-ml-neural.lock.txt to use LSTM/Transformer") from error
        raise


def predict(kind, model, windows, scalers, batch_size):
    if kind == "persistence":
        return persistence(windows)
    if kind == "mean-return":
        return windows.current[:, None, :] * (1 + model[None, :, :])
    if kind == "ridge":
        scaled = model.predict(windows.x.reshape(len(windows.x), -1))
    else:
        scaled = neural_module().predict_model(model, windows, batch_size)
    decoded = scalers["y"].inverse_transform(scaled.reshape(-1, len(STOCKS))).reshape(windows.y.shape)
    if scalers.get("target_representation", scalers.get("representation", "price")) == "relative":
        return windows.current[:, None, :] * (1 + decoded)
    return decoded


def fit_ridge(training, alpha):
    """Shared fixed solver for the single-split and walk-forward workflows."""
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-6)
    with threadpool_limits(limits=4):
        model.fit(training.x.reshape(len(training.x), -1), training.y_scaled.reshape(len(training.y), -1))
    return model


def evaluate(windows, predictions, output):
    output.mkdir(exist_ok=False)
    table, summary = score(windows, predictions)
    table.to_csv(output / "metrics.csv", index=False)
    prediction_table(windows, predictions).to_csv(output / "predictions.csv", index=False, float_format="%.10g")
    baseline_table, baseline_summary = score(windows, persistence(windows))
    baseline_table.to_csv(output / "persistence_metrics.csv", index=False)
    summary["persistence"] = baseline_summary
    write_json(output / "summary.json", summary)
    return summary


def provenance():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "exchange-calendars", "tensorflow", "keras", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {
        "versions": versions, "git_revision": revision.stdout.strip() or None,
        "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS",
        )},
        "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def train(args):
    config = DataConfig(args.lookback, args.horizon, args.train_end, args.validation_end,
                        args.feature_set, args.representation, args.input_representation, args.target_representation)
    config.validate()
    if args.alpha <= 0 or not np.isfinite(args.alpha):
        raise ValueError("alpha must be finite and positive")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("epochs/batch_size must be positive; patience must be nonnegative")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    source, output = args.data.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new run directory")
    frame, features, source_meta = load_frame(source, config.feature_set)
    dataset = make_dataset(frame, features, config)
    settings = {
        "alpha": args.alpha, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "patience": args.patience, "units": args.units,
        "learning_rate": args.learning_rate, "position_encoding": not args.no_position_encoding,
    }
    manifest = {
        "status": "incomplete", "model": args.model, "data_config": asdict(config),
        "settings": settings, "data_path": str(source), "data_sha256": digest(source),
        "source": source_meta, "dataset": dataset.metadata, **provenance(),
        "protocol": "Fit on train; early stopping/selection on validation only; test requires evaluate-test.",
    }
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    joblib.dump(dataset.scalers, output / "scalers.joblib")
    training, validation = dataset.partitions["train"], dataset.partitions["validation"]
    model = None
    if args.model == "mean-return":
        model = (training.y / training.current[:, None, :] - 1).mean(axis=0)
        joblib.dump(model, output / "model.joblib")
    elif args.model == "ridge":
        # Regularized multi-output linear baseline, replacing underdetermined OLS.
        model = fit_ridge(training, args.alpha)
        joblib.dump(model, output / "model.joblib")
    elif args.model in ("lstm", "transformer"):
        model, history = neural_module().train_model(args.model, training, validation, settings)
        model.save(output / "model.keras")
        write_json(output / "history.json", history)
    predictions = predict(args.model, model, validation, dataset.scalers, args.batch_size)
    summary = evaluate(validation, predictions, output / "validation")
    manifest["status"] = "complete"
    manifest["artifact_sha256"] = {
        p.name: digest(p) for p in output.iterdir() if p.suffix in (".joblib", ".keras")
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"run": str(output), "split": "validation", "summary": summary}, indent=2))


def evaluate_test(args):
    output = args.run.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Run did not finish successfully")
    if (output / "test").exists():
        raise ValueError("Test already evaluated for this run; inspect existing results")
    source = args.data.resolve() if args.data else Path(manifest["data_path"])
    if digest(source) != manifest["data_sha256"]:
        raise ValueError("Data hash mismatch: evaluate the exact training snapshot")
    for filename, expected in manifest["artifact_sha256"].items():
        if digest(output / filename) != expected:
            raise ValueError(f"Saved artifact hash mismatch: {filename}")
    config = DataConfig(**manifest["data_config"])
    frame, features, _ = load_frame(source, config.feature_set)
    # joblib is for locally produced, trusted run artifacts only.
    scalers = joblib.load(output / "scalers.joblib")
    windows = make_dataset(frame, features, config, scalers).partitions["test"]
    kind, model = manifest["model"], None
    if kind in ("ridge", "mean-return"):
        model = joblib.load(output / "model.joblib")
    elif kind in ("lstm", "transformer"):
        model = neural_module().keras.models.load_model(output / "model.keras", compile=False)
    predictions = predict(kind, model, windows, scalers, manifest["settings"]["batch_size"])
    summary = evaluate(windows, predictions, output / "test")
    write_json(output / "test" / "evaluation_manifest.json", {
        "training_manifest_sha256": digest(output / "manifest.json"), **provenance(),
    })
    print(json.dumps({"run": str(output), "split": "test", "summary": summary}, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Train and evaluate VALIDATION only")
    fit.add_argument("--data", type=Path, default=Path("api/data/total.csv"))
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--model", choices=("persistence", "mean-return", "ridge", "lstm", "transformer"), default="ridge")
    fit.add_argument("--feature-set", choices=("stock", "stock-econ"), default="stock")
    fit.add_argument("--representation", choices=("price", "relative"), default="price",
                     help="Joint input/target representation; relative uses origin-normalized stocks and future returns")
    fit.add_argument("--input-representation", choices=("price", "relative"),
                     help="Override only the input side of --representation")
    fit.add_argument("--target-representation", choices=("price", "relative"),
                     help="Override only the target side of --representation")
    fit.add_argument("--lookback", type=int, default=90)
    fit.add_argument("--horizon", type=int, default=7)
    fit.add_argument("--train-end", default="2021-12-30")
    fit.add_argument("--validation-end", default="2023-12-29")
    fit.add_argument("--alpha", type=float, default=10.0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--patience", type=int, default=5)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--units", type=int, default=64)
    fit.add_argument("--learning-rate", type=float, default=0.0001)
    fit.add_argument("--no-position-encoding", action="store_true", help="Transformer ablation only")
    fit.set_defaults(action=train)
    test = commands.add_parser("evaluate-test", help="Evaluate frozen run on held-out TEST; perform after selection")
    test.add_argument("--run", type=Path, required=True)
    test.add_argument("--data", type=Path, help="Same CSV snapshot at a different path; hash must match")
    test.set_defaults(action=evaluate_test)
    return result


def main():
    command_parser = parser()
    args = command_parser.parse_args()
    try:
        args.action(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        command_parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
