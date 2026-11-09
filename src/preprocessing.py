
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


# Features configuration
NUMERICAL_FEATURES = [
    "number of adults",
    "number of children",
    "number of weekend nights",
    "number of week nights",
    "car parking space",
    "lead time",
    "repeated",
    "P-C",
    "P-not-C",
    "average price",
    "special requests"
]

CATEGORICAL_FEATURES = [
    "type of meal",
    "room type",
    "market segment type"
]


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Stratification preserves the target class distribution
    in both sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Apply preprocessing to training and testing data.

    Steps:
        1. One-Hot Encoding for categorical features.
        2. Standard Scaling for numerical features.
        3. Label Encoding for the target variable.

    All preprocessing objects are fitted on training data only.
    """

    # Initialize preprocessing objects
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    scaler = StandardScaler()

    target_encoder = LabelEncoder()

    # Categorical Encoding
    X_train_encoded = encoder.fit_transform(
        X_train[CATEGORICAL_FEATURES]
    )

    X_test_encoded = encoder.transform(
        X_test[CATEGORICAL_FEATURES]
    )

    encoded_feature_names = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    # Convert encoded arrays into DataFrames
    X_train_encoded = pd.DataFrame(
        X_train_encoded,
        columns=encoded_feature_names,
        index=X_train.index
    )

    X_test_encoded = pd.DataFrame(
        X_test_encoded,
        columns=encoded_feature_names,
        index=X_test.index
    )

    # Numerical Scaling
    X_train_scaled = scaler.fit_transform(
        X_train[NUMERICAL_FEATURES]
    )

    X_test_scaled = scaler.transform(
        X_test[NUMERICAL_FEATURES]
    )

    # Convert scaled arrays into DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=NUMERICAL_FEATURES,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=NUMERICAL_FEATURES,
        index=X_test.index
    )

    # Combine numerical and encoded features
    X_train_processed = pd.concat(
        [X_train_scaled, X_train_encoded],
        axis=1
    )

    X_test_processed = pd.concat(
        [X_test_scaled, X_test_encoded],
        axis=1
    )

    # Target Encoding
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)

    return {
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "encoder": encoder,
        "scaler": scaler,
        "target_encoder": target_encoder
    }


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data and apply all preprocessing steps.

    Returns the processed datasets and fitted preprocessing objects.
    """

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    preprocessing_results = preprocess_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        **preprocessing_results
    }


def preprocess_new_data(data, encoder, scaler):
    """
    Apply the fitted preprocessing objects to new booking data.

    The encoder and scaler are used for transformation only.
    They are not fitted again.
    """

    # Categorical Encoding
    encoded_data = encoder.transform(
        data[CATEGORICAL_FEATURES]
    )

    encoded_feature_names = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    encoded_data = pd.DataFrame(
        encoded_data,
        columns=encoded_feature_names,
        index=data.index
    )

    # Numerical Scaling
    scaled_data = scaler.transform(
        data[NUMERICAL_FEATURES]
    )

    scaled_data = pd.DataFrame(
        scaled_data,
        columns=NUMERICAL_FEATURES,
        index=data.index
    )

    # Combine processed features
    processed_data = pd.concat(
        [scaled_data, encoded_data],
        axis=1
    )

    return processed_data
