
import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_random_forest_pipeline.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_booking(data, model=None):
    if model is None:
        model = load_model()

    predictions = model.predict(data)

    return predictions


def predict_booking_probability(data, model=None):
    if model is None:
        model = load_model()

    probabilities = model.predict_proba(data)

    probability_df = pd.DataFrame(
        probabilities,
        columns=model.classes_,
        index=data.index
    )

    return probability_df
