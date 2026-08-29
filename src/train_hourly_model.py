"""Train and save the hourly bike-rental prediction model."""

from pathlib import Path

from joblib import dump

from src.hourly_model import load_hourly_training_data, train_hourly_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hour.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "hourly_bike_rental_model.joblib"


def main() -> None:
    data = load_hourly_training_data(DATA_PATH)
    model = train_hourly_model(data)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    # Compression keeps the same trained model while making it small enough to deploy.
    dump(model, MODEL_PATH, compress=3)
    print(f"Hourly model saved to: {MODEL_PATH}")
    print(f"Training samples: {len(data)}")


if __name__ == "__main__":
    main()
