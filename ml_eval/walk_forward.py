"""Fixed-candidate quarterly expanding-window diagnostics on development years."""

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import DataConfig, Windows, load_frame, make_dataset
from .metrics import prediction_table, score, yearly_scores
from .run import digest, evaluate, fit_ridge, predict, provenance, write_json


ALPHA = 10000.0
RESERVED_START = pd.Timestamp("2024-01-01")
MODEL_NAMES = ("persistence", "mean-return-expanding", "ridge-expanding", "ridge-fixed")


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    evaluation_end: str


def quarterly_folds():
    result = []
    for period in pd.period_range("2022Q1", "2023Q4", freq="Q"):
        result.append(Fold(str(period), str((period.start_time - pd.Timedelta(days=1)).date()),
                           str(period.end_time.date())))
    return tuple(result)


def validate_folds(folds):
    if not folds:
        raise ValueError("At least one fold is required")
    if len({fold.name for fold in folds}) != len(folds):
        raise ValueError("Fold names must be unique")
    previous_end = None
    for fold in folds:
        if not fold.name or Path(fold.name).name != fold.name or fold.name in (".", ".."):
            raise ValueError("Fold name must be a single directory name")
        start, end = pd.Timestamp(fold.train_end), pd.Timestamp(fold.evaluation_end)
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError("Fold train cutoff must precede evaluation end")
        if end >= RESERVED_START:
            raise ValueError("2024 onward is reserved; this diagnostic must not evaluate it")
        if previous_end is not None and start != previous_end:
            raise ValueError("Expanding folds must be contiguous and ordered")
        previous_end = end


def concatenate_windows(parts):
    return Windows(**{field.name: np.concatenate([getattr(part, field.name) for part in parts])
                      for field in fields(Windows)})


def run_walk_forward(source: Path, output: Path, *, folds=None, lookback=90, horizon=7):
    folds = quarterly_folds() if folds is None else tuple(folds)
    validate_folds(folds)
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new study directory")
    manifest = {
        "status": "incomplete", "study": "quarterly-expanding-development-diagnostic",
        "data_path": str(source), "data_sha256": digest(source), "test_evaluated": False,
        "lookback": lookback, "horizon": horizon, "alpha": ALPHA,
        "input_representation": "relative", "target_representation": "relative",
        "models": list(MODEL_NAMES), "folds": [asdict(fold) for fold in folds],
        "selection_rule": "No candidate selection or tuning; fixed alpha=10000 from the preceding development study.",
        "limitations": [
            "Candidate was selected using these development years; this is retrospective stability analysis, not nested validation or an untouched holdout.",
            "Legacy snapshot is not point-in-time verified; release/vintage, prior backfill and constituent selection limitations remain.",
            "Overlapping target windows are dependent; fold win counts are descriptive, not significance tests.",
            "Quarter-end target windows that cross a refit boundary are excluded equally for every model.",
        ], **provenance(),
    }
    # Cut 2024+ values BEFORE preprocessing and slice again per fold in make_dataset.
    frame, features, source_metadata = load_frame(source, "stock", end_date=folds[-1].evaluation_end)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    window_parts, fold_labels, fold_results = [], [], []
    predictions = {name: [] for name in MODEL_NAMES}
    seen_targets = set()
    frozen_model = frozen_scalers = None
    frozen_fit_cutoff = None
    for fold in folds:
        print(f"{fold.name}: fit through {fold.train_end}, evaluate through {fold.evaluation_end}", flush=True)
        config = DataConfig(lookback, horizon, fold.train_end, fold.evaluation_end, representation="relative")
        dataset = make_dataset(frame, features, config, include_test=False)
        training, evaluation = dataset.partitions["train"], dataset.partitions["validation"]
        target_set = set(evaluation.target_dates.flatten())
        if seen_targets.intersection(target_set):
            raise ValueError("Evaluation targets overlap between folds")
        seen_targets.update(target_set)
        if training.target_dates.max() > np.datetime64(fold.train_end):
            raise ValueError("Training labels extend beyond refit cutoff")
        model = fit_ridge(training, ALPHA)
        mean_return = (training.y / training.current[:, None, :] - 1).mean(axis=0)
        if frozen_model is None:
            frozen_model, frozen_scalers = model, dataset.scalers
            frozen_fit_cutoff = dataset.metadata["scaler_fit_end"]
        directory = output / fold.name
        directory.mkdir()
        joblib.dump(model, directory / "ridge.joblib")
        joblib.dump(mean_return, directory / "mean-return.joblib")
        joblib.dump(dataset.scalers, directory / "scalers.joblib")
        # Relative inputs are origin-local and have no fitted x scaler, so their
        # values are identical for expanding and frozen models. Decode the latter
        # with the FIRST fold's y scaler, never the current fold's scaler.
        if dataset.scalers["x"] is not None or frozen_scalers["x"] is not None:
            raise ValueError("Frozen control requires origin-local relative inputs")
        fold_predictions = {
            "persistence": predict("persistence", None, evaluation, dataset.scalers, 32),
            "mean-return-expanding": predict("mean-return", mean_return, evaluation, dataset.scalers, 32),
            "ridge-expanding": predict("ridge", model, evaluation, dataset.scalers, 32),
            "ridge-fixed": predict("ridge", frozen_model, evaluation, frozen_scalers, 32),
        }
        summaries = {}
        for name, values in fold_predictions.items():
            summary = evaluate(evaluation, values, directory / name)
            summaries[name] = {key: value for key, value in summary.items() if key != "persistence"}
            predictions[name].append(values)
        fold_manifest = {
            **asdict(fold), "dataset": dataset.metadata, "data_config": asdict(config),
            "fixed_control_fit_end": frozen_fit_cutoff,
            "fixed_control_artifact_directory": folds[0].name,
            "artifact_sha256": {p.name: digest(p) for p in directory.glob("*.joblib")},
            "summaries": summaries,
        }
        write_json(directory / "manifest.json", fold_manifest)
        fold_results.append(fold_manifest)
        window_parts.append(evaluation)
        fold_labels.extend([fold.name] * len(evaluation.y))
    combined = concatenate_windows(window_parts)
    combined_summaries = {}
    aggregate_directory = output / "aggregate"
    aggregate_directory.mkdir()
    for name in MODEL_NAMES:
        values = np.concatenate(predictions[name])
        table, summary = score(combined, values)
        table.to_csv(aggregate_directory / f"{name}-metrics.csv", index=False)
        exported = prediction_table(combined, values)
        exported.insert(0, "fold", np.repeat(fold_labels, horizon * len(features)))
        exported.to_csv(aggregate_directory / f"{name}-predictions.csv", index=False, float_format="%.10g")
        summary["by_target_year"] = yearly_scores(combined, values)
        summary["fold_mape_wins_vs_persistence"] = sum(
            fold["summaries"][name]["macro_mape_pct"] < fold["summaries"]["persistence"]["macro_mape_pct"]
            for fold in fold_results
        )
        combined_summaries[name] = summary
    if digest(source) != manifest["data_sha256"]:
        raise ValueError("Source snapshot changed during study")
    result = {
        "evaluation_split": "walk-forward-development", "test_evaluated": False,
        "fold_count": len(folds), "evaluated_origin_count": len(combined.y),
        "source": source_metadata, "models": combined_summaries, "folds": fold_results,
        "limitations": manifest["limitations"],
    }
    write_json(output / "comparison.json", result)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("api/data/total.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_walk_forward(args.data, args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Completed {result['fold_count']} folds / {result['evaluated_origin_count']} origins. See {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
