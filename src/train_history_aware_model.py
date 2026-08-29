"""Train and save the history-aware hourly prediction model."""

from pathlib import Path

from joblib import dump

from src.history_aware_model import prepare_history_aware_data, train_history_aware_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hour.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "history_aware_hourly_bike_rental_model.joblib"


def main() -> None:
    data = prepare_history_aware_data(DATA_PATH)
    model = train_history_aware_model(data)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    dump(model, MODEL_PATH, compress=3)
    print(f"History-aware hourly model saved to: {MODEL_PATH}")
    print(f"Training samples: {len(data)}")


if __name__ == "__main__":
    main()
