"""Train and save the tuned random-forest model for the Streamlit app."""

from pathlib import Path

from joblib import dump

from src.model import train_model, load_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "day.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "bike_rental_model.joblib"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Download the Kaggle dataset and place day.csv in data/."
        )

    data = load_training_data(DATA_PATH)
    model = train_model(data)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"Model trained with {len(data)} daily records.")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
