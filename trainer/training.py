import logging
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from google.cloud import storage
from io import BytesIO


def load_csv_data(bucket_name, blob_path):
    """Load data from GCS bucket
    Args:
        bucket_name (str): Name of the GCS bucket
        blob_path (str): Path to the blob in the bucket
    Returns:
        pd.DataFrame: DataFrame containing the loaded data
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return pd.read_csv(BytesIO(blob.download_as_bytes()))


def Encoder(df: pd.DataFrame) -> pd.DataFrame:
    columnsToEncode = list(df.select_dtypes(include=["category", "object"]))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except ValueError:
            logging.info("Error encoding " + feature)
    return df


def save_model(model, filename):
    """Save the model to a file
    Args:
        model: The model to save
        filename (str): The filename to save the model to
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    """Load the model from a file
    Args:
        filename (str): The filename to load the model from
    Returns:
        The loaded model
    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def generate_ds(df):
    """Generate the dataset for training
    Args:
        df (pd.DataFrame): The DataFrame containing the data
    Returns:
        tuple: A tuple containing the features and target variable
    """
    X = df[
        [
            "dist_to_restaurant",
            "Hdist_to_restaurant",
            "avg_Hdist_to_restaurants",
            "date_day_number",
            "restaurant_id",
            "Five_Clusters_embedding",
            "h3_index",
            "date_hour_number",
            "restaurants_per_index",
        ]
    ]
    y = df[["orders_busyness_by_h3_hour"]]

    return X, y


def combine_ds(X, y):
    return pd.concat((X, y), axis=1)
