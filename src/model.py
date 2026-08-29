"""Training utilities for the bike-sharing rental prediction model."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURES = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

BEST_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": 0.8,
    "max_depth": 12,
    "random_state": 42,
    "n_jobs": 1,
}


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load the daily rental dataset and validate the required columns."""
    data = pd.read_csv(data_path)
    required_columns = set(FEATURES + ["cnt"])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    return data


def train_model(data: pd.DataFrame) -> RandomForestRegressor:
    """Train the tuned random-forest model using all available historical data."""
    model = RandomForestRegressor(**BEST_PARAMS)
    model.fit(data[FEATURES], data["cnt"])
    return model
