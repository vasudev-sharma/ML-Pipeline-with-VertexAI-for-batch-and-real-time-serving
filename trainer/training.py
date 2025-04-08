import logging
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from google.cloud import storage
from io import BytesIO

def load_data(bucket_name, blob_path):
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
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def generate_ds(df):
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

def combine_ds_and_save(X, y, filename):
    combine_ds = pd.concat((X, y), axis=1)

    try:
        combine_ds.to_csv(filename)
    except Exception as e:
        print("Unable to save the data csv file")