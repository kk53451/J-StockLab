"""Trading-session alignment and target-disjoint temporal partitions."""

from dataclasses import dataclass
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


STOCKS = (
    "Toyota", "SoftBank Group", "Mitsubishi UFJ Financial", "Sony Group",
    "Hitachi", "Fast Retailing", "SMFG", "Nintendo", "Tokyo Electron",
    "Advantest", "Mitsubishi Heavy Ind", "Mitsubishi Corp", "Keyence",
    "Chugai Pharma", "ITOCHU", "Mizuho Financial", "NTT", "Mitsui & Co",
    "Recruit Holdings", "Tokio Marine",
)


@dataclass(frozen=True)
class DataConfig:
    lookback: int = 90
    horizon: int = 7
    train_end: str = "2021-12-30"
    validation_end: str = "2023-12-29"
    feature_set: str = "stock"

    def validate(self):
        if self.lookback < 1 or self.horizon < 1:
            raise ValueError("lookback and horizon must be positive")
        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.validation_end):
            raise ValueError("train_end must precede validation_end")
        if self.feature_set not in ("stock", "stock-econ"):
            raise ValueError("feature_set must be stock or stock-econ")


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    y_scaled: np.ndarray
    current: np.ndarray
    origins: np.ndarray
    target_dates: np.ndarray


@dataclass
class Dataset:
    partitions: dict[str, Windows]
    scalers: dict
    metadata: dict


def load_frame(path: Path, feature_set: str) -> tuple[pd.DataFrame, list[str], dict]:
    frame = pd.read_csv(path)
    if "날짜" not in frame:
        raise ValueError("CSV must contain 날짜")
    dates = pd.to_datetime(frame.pop("날짜"), errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be non-null and unique")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Expected date-only daily rows")
    frame.index = pd.DatetimeIndex(dates)
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("CSV is empty")
    missing_columns = set(STOCKS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing stock columns: {sorted(missing_columns)}")

    # Explicit bounds avoid the calendar package's moving default date range.
    # A CSV may begin/end on a holiday; give the calendar sessions on both
    # sides so sessions_in_range does not reject a non-session boundary.
    calendar = xcals.get_calendar("XTKS", start=frame.index.min() - pd.Timedelta(days=31),
                                  end=frame.index.max() + pd.Timedelta(days=31))
    sessions = calendar.sessions_in_range(frame.index.min(), frame.index.max())
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    if sessions.empty:
        raise ValueError("CSV contains no trading sessions")
    missing_sessions = sessions.difference(frame.index)
    if len(missing_sessions):
        raise ValueError(f"Missing trading sessions; recollect data: {missing_sessions[:5].tolist()}")
    source_rows = len(frame)
    frame = frame.loc[sessions].copy()
    features = list(STOCKS)
    metadata = {
        "calendar": "XTKS", "source_rows": source_rows,
        "trading_sessions": len(frame), "removed_non_sessions": source_rows - len(frame),
        "economic_lag_sessions": 0,
        "limitations": [
            "Legacy snapshot: prior backfill, revisions, adjusted prices and constituent selection are not point-in-time verified.",
            "A calendar session does not prove each stored value is an original unfilled observation.",
        ],
    }
    if feature_set == "stock-econ":
        economics = [column for column in frame.columns if column not in STOCKS]
        if not economics:
            raise ValueError("stock-econ requires economic columns")
        # A one-session lag avoids same-date US close usage. It does NOT repair
        # macro release timing, historical revisions, or prior legacy backfill.
        frame[economics] = frame[economics].apply(pd.to_numeric, errors="raise").ffill().shift(1)
        features += economics
        valid = frame[economics].notna().all(axis=1)
        if not valid.any():
            raise ValueError("No complete economic observations")
        frame = frame.loc[valid[valid].index[0]:].copy()
        metadata["economic_lag_sessions"] = 1
        metadata["limitations"].append(
            "EXPLORATORY: lagged legacy economics are not release-date/vintage aligned; do not claim leakage-free performance."
        )
    frame = frame[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite feature/target values; do not backfill or silently skip trading sessions")
    if (frame[list(STOCKS)] <= 0).any().any():
        raise ValueError("Stock prices must be positive")
    metadata.update(usable_sessions=len(frame), start=str(frame.index.min().date()), end=str(frame.index.max().date()))
    return frame, features, metadata


def make_dataset(frame: pd.DataFrame, features: list[str], config: DataConfig,
                 scalers: dict | None = None) -> Dataset:
    config.validate()
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Frame must have sorted unique dates")
    train_end, validation_end = pd.Timestamp(config.train_end), pd.Timestamp(config.validation_end)
    training = frame.loc[frame.index <= train_end]
    if len(training) < config.lookback + config.horizon:
        raise ValueError("Not enough training sessions")
    if scalers is None:
        scalers = {
            "features": features, "stocks": list(STOCKS),
            "x": MinMaxScaler().fit(training[features]),
            "y": MinMaxScaler().fit(training[list(STOCKS)]),
        }
    if scalers["features"] != features or scalers["stocks"] != list(STOCKS):
        raise ValueError("Saved scaler schema does not match dataset")
    x_values = scalers["x"].transform(frame[features]).astype(np.float32)
    y_values = frame[list(STOCKS)].to_numpy(dtype=np.float64)
    y_scaled = scalers["y"].transform(frame[list(STOCKS)]).astype(np.float32)

    # Origin is the LAST available input session. Day1 is immediately next.
    origins = np.arange(config.lookback - 1, len(frame) - config.horizon)
    inputs = origins[:, None] - np.arange(config.lookback - 1, -1, -1)
    targets = origins[:, None] + np.arange(1, config.horizon + 1)
    dates = frame.index.to_numpy()
    target_dates = dates[targets]
    masks = {
        "train": target_dates[:, -1] <= train_end.to_datetime64(),
        "validation": (target_dates[:, 0] > train_end.to_datetime64()) & (target_dates[:, -1] <= validation_end.to_datetime64()),
        "test": target_dates[:, 0] > validation_end.to_datetime64(),
    }
    partitions, split_metadata = {}, {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No {name} windows; adjust cutoffs/lookback/horizon")
        partitions[name] = Windows(
            x=x_values[inputs[mask]], y=y_values[targets[mask]],
            y_scaled=y_scaled[targets[mask]], current=y_values[origins[mask]],
            origins=dates[origins[mask]], target_dates=target_dates[mask],
        )
        split_metadata[name] = {
            "samples": int(mask.sum()),
            "first_origin": str(dates[origins[mask][0]])[:10],
            "last_origin": str(dates[origins[mask][-1]])[:10],
            "first_target": str(target_dates[mask][0, 0])[:10],
            "last_target": str(target_dates[mask][-1, -1])[:10],
        }
    return Dataset(partitions, scalers, {
        "partitions": split_metadata,
        "purged_boundary_windows": int(len(origins) - sum(mask.sum() for mask in masks.values())),
        "scaler_fit_start": str(training.index.min().date()),
        "scaler_fit_end": str(training.index.max().date()),
        "features": features,
    })
