"""Training utilities for the hourly bike-rental prediction model."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


HOURLY_FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]

HOURLY_MODEL_PARAMS = {"n_estimators": 150, "random_state": 42, "n_jobs": 1}


def load_hourly_training_data(data_path: Path) -> pd.DataFrame:
    """Load hourly data and validate the fields used for training."""
    data = pd.read_csv(data_path)
    missing_columns = set(HOURLY_FEATURES + ["cnt"]).difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    return data


def train_hourly_model(data: pd.DataFrame) -> RandomForestRegressor:
    """Fit the chosen baseline random forest using all historical hourly data."""
    model = RandomForestRegressor(**HOURLY_MODEL_PARAMS)
    model.fit(data[HOURLY_FEATURES], data["cnt"])
    return model
