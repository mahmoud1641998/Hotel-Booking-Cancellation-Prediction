import pandas as pd
import pytest

from src.preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_data,
    preprocess_new_data
)


@pytest.fixture
def sample_data():
    """Create a small dataset for preprocessing tests."""

    n_samples = 30

    data = {}

    # Numerical features
    for i, feature in enumerate(NUMERICAL_FEATURES):
        data[feature] = [j + i for j in range(n_samples)]

    # Categorical features
    data["type of meal"] = [
        ["Meal Plan 1", "Meal Plan 2", "Not Selected"][j % 3]
        for j in range(n_samples)
    ]

    data["room type"] = [
        f"Room_Type{(j % 7) + 1}"
        for j in range(n_samples)
    ]

    data["market segment type"] = [
        ["Online", "Offline", "Corporate", "Complementary", "Aviation"][j % 5]
        for j in range(n_samples)
    ]

    X = pd.DataFrame(data, index=range(100, 100 + n_samples))

    y = pd.Series(
        ["Canceled", "Not_Canceled"] * 15,
        index=X.index
    )

    return X, y


def test_prepare_data_split(sample_data):
    """Test dataset splitting and stratification."""

    X, y = sample_data

    result = prepare_data(X, y, test_size=0.2, random_state=42)

    assert len(result["X_train"]) == 24
    assert len(result["X_test"]) == 6

    assert len(result["y_train"]) == 24
    assert len(result["y_test"]) == 6

    # Check stratification
    assert result["y_train"].value_counts()["Canceled"] == 12
    assert result["y_test"].value_counts()["Canceled"] == 3


def test_processed_data_shape_and_values(sample_data):
    """Test processed feature dimensions and missing values."""

    X, y = sample_data

    result = prepare_data(X, y)

    expected_features = (
        len(NUMERICAL_FEATURES)
        + sum(len(categories) for categories in result["encoder"].categories_)
    )

    assert result["X_train_processed"].shape == (24, expected_features)
    assert result["X_test_processed"].shape == (6, expected_features)

    assert result["X_train_processed"].isnull().sum().sum() == 0
    assert result["X_test_processed"].isnull().sum().sum() == 0

    assert result["X_train_processed"].index.equals(result["X_train"].index)
    assert result["X_test_processed"].index.equals(result["X_test"].index)


def test_target_encoding(sample_data):
    """Test target encoding and class mapping."""

    X, y = sample_data

    result = prepare_data(X, y)

    target_encoder = result["target_encoder"]

    assert target_encoder.transform(["Canceled"])[0] == 0
    assert target_encoder.transform(["Not_Canceled"])[0] == 1

    assert len(result["y_train_encoded"]) == 24
    assert len(result["y_test_encoded"]) == 6


def test_preprocess_new_data(sample_data):
    """Test preprocessing of new data using fitted objects."""

    X, y = sample_data

    result = prepare_data(X, y)

    new_data = result["X_test"]

    processed_data = preprocess_new_data(
        new_data,
        result["encoder"],
        result["scaler"]
    )

    pd.testing.assert_frame_equal(
        processed_data,
        result["X_test_processed"]
    )


def test_preprocess_new_data_with_unknown_category(sample_data):
    """Test that unknown categorical values are handled correctly."""

    X, y = sample_data

    result = prepare_data(X, y)

    new_data = result["X_test"].copy()

    new_data.iloc[0, new_data.columns.get_loc("type of meal")] = "Unknown Meal"

    processed_data = preprocess_new_data(
        new_data,
        result["encoder"],
        result["scaler"]
    )

    assert processed_data.shape[0] == len(new_data)
    assert processed_data.shape[1] == result["X_test_processed"].shape[1]
    assert processed_data.isnull().sum().sum() == 0