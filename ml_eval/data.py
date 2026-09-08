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
    representation: str = "price"
    input_representation: str | None = None
    target_representation: str | None = None

    @property
    def input_mode(self):
        return self.input_representation or self.representation

    @property
    def target_mode(self):
        return self.target_representation or self.representation

    def validate(self):
        if self.lookback < 1 or self.horizon < 1:
            raise ValueError("lookback and horizon must be positive")
        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.validation_end):
            raise ValueError("train_end must precede validation_end")
        if self.feature_set not in ("stock", "stock-econ"):
            raise ValueError("feature_set must be stock or stock-econ")
        if self.representation not in ("price", "relative"):
            raise ValueError("representation must be price or relative")
        for mode in (self.input_representation, self.target_representation):
            if mode is not None and mode not in ("price", "relative"):
                raise ValueError("input/target representation must be price or relative")
        if "relative" in (self.input_mode, self.target_mode) and self.feature_set != "stock":
            raise ValueError("relative representation currently requires stock-only features")


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
    y_values = frame[list(STOCKS)].to_numpy(dtype=np.float64)

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
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No {name} windows; adjust cutoffs/lookback/horizon")
    current = y_values[origins]
    target_prices = y_values[targets]
    relative_input = config.input_mode == "relative"
    relative_target = config.target_mode == "relative"
    if relative_input and features != list(STOCKS):
        raise ValueError("relative representation requires stock columns in canonical order")
    model_targets = target_prices / current[:, None, :] - 1 if relative_target else target_prices
    if scalers is None:
        scalers = {
            "features": features, "stocks": list(STOCKS), "representation": config.representation,
            "input_representation": config.input_mode, "target_representation": config.target_mode,
            # Relative inputs are dimensionless and origin-local; no fitted input scaler.
            "x": None if relative_input else MinMaxScaler().fit(training[features]),
            "y": MinMaxScaler().fit(model_targets[masks["train"]].reshape(-1, len(STOCKS)))
                 if relative_target else MinMaxScaler().fit(training[list(STOCKS)]),
        }
    if (scalers["features"] != features or scalers["stocks"] != list(STOCKS)
            or scalers.get("input_representation", scalers.get("representation", "price")) != config.input_mode
            or scalers.get("target_representation", scalers.get("representation", "price")) != config.target_mode):
        raise ValueError("Saved scaler schema/representation does not match dataset")
    if relative_input:
        x_windows = (y_values[inputs] / current[:, None, :] - 1).astype(np.float32)
    else:
        x_windows = scalers["x"].transform(frame[features]).astype(np.float32)[inputs]
    if relative_target:
        target_scaled = scalers["y"].transform(model_targets.reshape(-1, len(STOCKS))).reshape(model_targets.shape)
    else:
        target_scaled = scalers["y"].transform(frame[list(STOCKS)])[targets]
    target_scaled = target_scaled.astype(np.float32)
    partitions, split_metadata = {}, {}
    for name, mask in masks.items():
        partitions[name] = Windows(
            x=x_windows[mask], y=target_prices[mask],
            y_scaled=target_scaled[mask], current=current[mask],
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
        "representation": config.representation,
        "input_representation": config.input_mode,
        "target_representation": config.target_mode,
        "target_definition": "price/origin_price - 1" if relative_target else "price",
    })
