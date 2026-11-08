
from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "booking.csv"


def load_data(file_path=DATASET_PATH):
    """
    Load the hotel booking dataset from a CSV file.

    Parameters:
        file_path: Path to the CSV file.

    Returns:
        pandas DataFrame.
    """

    df = pd.read_csv(file_path)

    return df
