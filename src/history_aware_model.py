"""Training utilities for hourly prediction with historical demand features."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.hourly_model import HOURLY_FEATURES


HISTORY_FEATURES = ["lag_1", "lag_24", "lag_168", "rolling_mean_24"]
HISTORY_AWARE_FEATURES = HOURLY_FEATURES + HISTORY_FEATURES
HISTORY_AWARE_MODEL_PARAMS = {"n_estimators": 150, "random_state": 42, "n_jobs": 1}


def prepare_history_aware_data(data_path: Path) -> pd.DataFrame:
    """Create past-only demand features from the chronological hourly dataset."""
    data = pd.read_csv(data_path, parse_dates=["dteday"])
    required = set(HOURLY_FEATURES + ["cnt", "dteday"])
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True).copy()
    data["lag_1"] = data["cnt"].shift(1)
    data["lag_24"] = data["cnt"].shift(24)
    data["lag_168"] = data["cnt"].shift(168)
    data["rolling_mean_24"] = data["cnt"].shift(1).rolling(24).mean()
    return data.dropna(subset=HISTORY_FEATURES).reset_index(drop=True)


def train_history_aware_model(data: pd.DataFrame) -> RandomForestRegressor:
    """Fit the history-aware hourly random forest on all available data."""
    model = RandomForestRegressor(**HISTORY_AWARE_MODEL_PARAMS)
    model.fit(data[HISTORY_AWARE_FEATURES], data["cnt"])
    return model
