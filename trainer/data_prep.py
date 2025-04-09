import logging
from io import StringIO
from typing import Tuple

import pandas as pd


def read_data(filename: str) -> pd.DataFrame:
    """Read the data from the given filename
    Args:
        filename (str): The filename to read
    Returns:
        pd.DataFrame: The processed DataFrame
    Raises:
        FileNotFoundError: If the file is not found
    """
    # TODO: Add docstrings and Type hints
    try:
        logging.info(f"Reading from filename: {filename}")
        df = pd.read_csv(filename)
        df.dropna(axis=0, inplace=True)
        return df
    except FileNotFoundError as e:
        raise e


def get_unique_restauraunts(processed_df) -> Tuple[pd.DataFrame, int]:
    """Get unique restaurants from the processed DataFrame
    Args:
        processed_df (pd.DataFrame): The processed DataFrame
    Returns:
        Tuple[pd.DataFrame, int]: A tuple containing the processed DataFrame and the number of unique restaurants
    """
    # TODO: Add type hints and docstrings

    # unique restaurants
    restaurants_ids = {}
    for a, b in zip(processed_df.restaurant_lat, processed_df.restaurant_lon):
        id = "{}_{}".format(a, b)
        restaurants_ids[id] = {"lat": a, "lon": b}
    for i, key in enumerate(restaurants_ids.keys()):
        restaurants_ids[key]["id"] = i

    # labeling of restaurants
    processed_df["restaurant_id"] = [
        restaurants_ids["{}_{}".format(a, b)]["id"]
        for a, b in zip(processed_df.restaurant_lat, processed_df.restaurant_lon)
    ]

    # number of unique restaurants
    logging.info(f"Number of unique restaurants are {len(restaurants_ids)}")
    print("The unique restarants are", len(restaurants_ids))

    return processed_df, restaurants_ids


# TODO: Change fn name
def get_restaurants_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Get the restaurants DataFrame and the number of unique restaurants"""

    # Unique restaurants
    restaurants_df, num_restauraunts_ids = get_unique_restauraunts(df)
    return restaurants_df, num_restauraunts_ids
