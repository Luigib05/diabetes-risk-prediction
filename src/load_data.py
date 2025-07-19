import os
import pandas as pd


def load_diabetes_data(filepath: str) -> pd.DataFrame:
    """
    Load the diabetes dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is an error parsing the CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")

    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file.")
