import os
import pandas as pd


def load_heart_data(data_dir: str = "data", filename: str = "heart.csv") -> pd.DataFrame:
    """
    Load the heart disease dataset from the specified directory.

    Parameters
    ----------
    data_dir : str
        Directory where the CSV file is stored.
    filename : str
        Name of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Please ensure the file exists.")
    df = pd.read_csv(path)
    return df
