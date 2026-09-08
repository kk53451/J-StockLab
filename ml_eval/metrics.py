"""Price errors and exact down/flat/up direction agreement; no 'accuracy=100-MAPE'."""

import numpy as np
import pandas as pd

from dataclasses import fields

from .data import STOCKS, Windows


def persistence(windows: Windows) -> np.ndarray:
    return np.broadcast_to(windows.current[:, None, :], windows.y.shape).copy()


def score(windows: Windows, predictions: np.ndarray) -> tuple[pd.DataFrame, dict]:
    if predictions.shape != windows.y.shape or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite and have shape (samples, horizon, stocks)")
    actual = windows.y
    current = windows.current[:, None, :]
    error = predictions - actual
    baseline_error = current - actual
    rows = []
    for day in range(actual.shape[1]):
        for stock_idx, stock in enumerate(STOCKS):
            residual = error[:, day, stock_idx]
            baseline_mae = float(np.abs(baseline_error[:, day, stock_idx]).mean())
            mae = float(np.abs(residual).mean())
            rows.append({
                "stock": stock, "horizon_sessions": day + 1, "samples": len(actual),
                "mae": mae, "rmse": float(np.sqrt(np.square(residual).mean())),
                "mape_pct": float((np.abs(residual) / actual[:, day, stock_idx]).mean() * 100),
                "direction_agreement_pct": float((
                    np.sign(predictions[:, day, stock_idx] - windows.current[:, stock_idx]) ==
                    np.sign(actual[:, day, stock_idx] - windows.current[:, stock_idx])
                ).mean() * 100),
                "persistence_mae": baseline_mae,
                "mae_skill_vs_persistence": 1 - mae / baseline_mae if baseline_mae > 0 else None,
            })
    table = pd.DataFrame(rows)
    skills = table["mae_skill_vs_persistence"].dropna()
    summary = {
        "origin_count": len(actual), "stock_count": len(STOCKS), "horizon_sessions": actual.shape[1],
        "macro_mape_pct": float(table["mape_pct"].mean()),
        "macro_direction_agreement_pct": float(table["direction_agreement_pct"].mean()),
        "macro_mae_skill_vs_persistence": float(skills.mean()) if len(skills) else None,
        "non_positive_prediction_count": int((predictions <= 0).sum()),
        "definitions": {
            "aggregation": "Equal weight per stock and horizon; overlapping origins are not independent observations.",
            "direction": "Exact sign(predicted-origin) == sign(actual-origin); flat is a third class.",
            "skill": "1 - model_MAE / persistence_MAE per stock/horizon; positive beats persistence; zero denominator excluded.",
            "prices": "Input snapshot price units; MAE/RMSE retained per stock, not averaged across different price scales.",
        },
    }
    return table, summary


def prediction_table(windows: Windows, predictions: np.ndarray) -> pd.DataFrame:
    n, horizon, stocks = predictions.shape
    current = np.broadcast_to(windows.current[:, None, :], predictions.shape)
    return pd.DataFrame({
        "origin_date": np.repeat(windows.origins, horizon * stocks),
        "target_date": np.repeat(windows.target_dates.reshape(-1), stocks),
        "stock": np.tile(STOCKS, n * horizon),
        "horizon_sessions": np.tile(np.repeat(np.arange(1, horizon + 1), stocks), n),
        "origin_price": current.reshape(-1), "actual_price": windows.y.reshape(-1),
        "predicted_price": predictions.reshape(-1),
        "predicted_return_pct": ((predictions / current - 1) * 100).reshape(-1),
    })


def yearly_scores(windows: Windows, predictions: np.ndarray) -> dict:
    """Score only windows whose entire target horizon lies in one year."""
    if predictions.shape != windows.y.shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid prediction shape or values")
    first = pd.DatetimeIndex(windows.target_dates[:, 0]).year.to_numpy()
    last = pd.DatetimeIndex(windows.target_dates[:, -1]).year.to_numpy()
    result = {"excluded_cross_year_windows": int((first != last).sum()), "years": {}}
    for year in sorted(set(first)):
        mask = (first == year) & (last == year)
        if not mask.any():
            continue
        subset = Windows(**{field.name: getattr(windows, field.name)[mask] for field in fields(Windows)})
        _, model_summary = score(subset, predictions[mask])
        _, baseline_summary = score(subset, persistence(subset))
        result["years"][str(year)] = {"model": model_summary, "persistence": baseline_summary}
    return result
