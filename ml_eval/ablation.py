"""Run the predefined 2x2 representation / Ridge regularization validation study."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .data import DataConfig, load_frame, make_dataset
from .metrics import persistence, prediction_table, yearly_scores
from .run import digest, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(output: Path):
    plan = read_json(output / "plan.json")
    source = Path(plan["data"])
    if digest(source) != plan["data_sha256"]:
        raise ValueError("Study data changed")
    config = DataConfig(**plan["data_config"])
    frame, features, source_metadata = load_frame(source, config.feature_set)
    dataset = make_dataset(frame, features, config)
    windows = dataset.partitions["validation"]
    keys = ["origin_date", "target_date", "stock", "horizon_sessions"]
    expected_keys = prediction_table(windows, persistence(windows))[keys].astype(str)
    rows = []
    reference = None
    for candidate in plan["candidates"]:
        run = output / candidate["name"]
        manifest = read_json(run / "manifest.json")
        if manifest["status"] != "complete" or manifest["data_sha256"] != plan["data_sha256"]:
            raise ValueError(f"Incomplete or different data snapshot: {run}")
        if (run / "test").exists():
            raise ValueError("This study must not evaluate test data")
        common = {key: manifest[key] for key in ("source_sha256", "versions")}
        if reference is None:
            reference = common
        if reference != common or manifest["dataset"]["partitions"] != dataset.metadata["partitions"]:
            raise ValueError(f"Incomparable code/environment/partitions: {run}")
        predictions = pd.read_csv(run / "validation/predictions.csv")
        if not predictions[keys].astype(str).equals(expected_keys):
            raise ValueError(f"Prediction dates or order do not match: {run}")
        values = predictions.predicted_price.to_numpy().reshape(windows.y.shape)
        summary = read_json(run / "validation/summary.json")
        rows.append({
            **candidate, "run_directory": str(run), "settings": manifest["settings"],
            "summary": {key: value for key, value in summary.items() if key != "persistence"},
            "by_target_year": yearly_scores(windows, values),
        })
    best = min((row for row in rows if row["model"] == "ridge"), key=lambda row: row["summary"]["macro_mape_pct"])
    result = {
        "evaluation_split": "validation", "test_evaluated": False,
        "selection_rule": plan["selection_rule"], "data_sha256": plan["data_sha256"],
        "data_config": plan["data_config"], "source": source_metadata, **reference,
        "partitions": dataset.metadata["partitions"], "best_ridge_candidate": best["name"],
        "persistence": summary["persistence"], "runs": rows,
        "limitations": [
            "Development validation has already informed earlier experiments; this is not a fresh holdout claim.",
            "Year slices use the same fixed model, not walk-forward refits; cross-year target windows excluded only from slices.",
            "Equal numeric alpha does not imply equal effective regularization under different input scales.",
        ],
    }
    write_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("api/data/total.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, source = args.output.resolve(), args.data.resolve()
    if output.exists():
        parser.error("Study output exists; use a new directory")
    candidates = [
        {"name": f"{x}-{y}-a{alpha}", "model": "ridge", "input": x, "target": y, "alpha": alpha}
        for x in ("price", "relative") for y in ("price", "relative") for alpha in (100, 1000, 10000)
    ]
    candidates.append({"name": "mean-return", "model": "mean-return", "input": "price", "target": "relative", "alpha": 100})
    plan = {
        "data": str(source), "data_sha256": digest(source),
        "data_config": {"lookback": 90, "horizon": 7, "train_end": "2021-12-30", "validation_end": "2023-12-29", "feature_set": "stock"},
        "selection_rule": "Minimum overall validation macro MAPE over 12 Ridge candidates; compare against persistence and training mean return; year slices are diagnostics only.",
        "candidates": candidates,
    }
    output.mkdir(parents=True)
    # Freeze candidate list BEFORE any fitting/evaluation.
    write_json(output / "plan.json", plan)
    for candidate in candidates:
        command = [
            sys.executable, "-m", "ml_eval.run", "train", "--data", str(source),
            "--output", str(output / candidate["name"]), "--model", candidate["model"],
            "--input-representation", candidate["input"], "--target-representation", candidate["target"],
            "--alpha", str(candidate["alpha"]),
        ]
        for key, value in plan["data_config"].items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        print("Training", candidate["name"], flush=True)
        with (output / f"{candidate['name']}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = summarize(output)
    print(json.dumps({"best_ridge_candidate": result["best_ridge_candidate"], "comparison": str(output / "comparison.json")}, indent=2))


if __name__ == "__main__":
    main()
