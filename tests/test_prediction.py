
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data
from src.predict import (
    load_model,
    predict_booking,
    predict_booking_probability
)


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def sample_data():
    df = load_data()

    X = df.drop(
        columns=[
            "Booking_ID",
            "date of reservation",
            "booking status"
        ]
    )

    return X.head(5)


def test_load_model(model):
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_booking_shape(model, sample_data):
    predictions = predict_booking(sample_data, model)

    assert len(predictions) == len(sample_data)
    assert isinstance(predictions, np.ndarray)


def test_predict_booking_valid_classes(model, sample_data):
    predictions = predict_booking(sample_data, model)

    valid_classes = ["Canceled", "Not_Canceled"]

    assert all(
        prediction in valid_classes
        for prediction in predictions
    )


def test_predict_booking_probability_shape(model, sample_data):
    probabilities = predict_booking_probability(
        sample_data,
        model
    )

    assert probabilities.shape == (len(sample_data), 2)
    assert isinstance(probabilities, pd.DataFrame)

    assert list(probabilities.columns) == list(model.classes_)


def test_predict_booking_probability_values(model, sample_data):
    probabilities = predict_booking_probability(
        sample_data,
        model
    )

    assert probabilities.notna().all().all()
    assert (probabilities >= 0).all().all()
    assert (probabilities <= 1).all().all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1).values,
        np.ones(len(sample_data))
    )


def test_prediction_without_passing_model(sample_data):
    predictions = predict_booking(sample_data)

    assert len(predictions) == len(sample_data)


def test_probability_without_passing_model(sample_data):
    probabilities = predict_booking_probability(sample_data)

    assert probabilities.shape == (len(sample_data), 2)
