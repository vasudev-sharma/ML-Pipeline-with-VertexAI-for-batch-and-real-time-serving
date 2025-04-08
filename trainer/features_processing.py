# features.py


import collections
from math import radians
import pandas as pd
from typing import Dict
import numpy as np
from sklearn.preprocessing import LabelEncoder


# calc. eucl. distances to restaurants arrays
def calc_dist(p1x, p1y, p2x, p2y):
    p1 = (p2x - p1x) ** 2
    p2 = (p2y - p1y) ** 2
    dist = np.sqrt(p1 + p2)
    return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else dist


# calc. avg. distance to restaurants
def avg_dist_to_restaurants(courier_lat, courier_lon, restaurants_ids):
    return np.mean(
        [
            calc_dist(v["lat"], v["lon"], courier_lat, courier_lon)
            for v in restaurants_ids.values()
        ]
    )


def calc_haversine_dist(lat1, lon1, lat2, lon2):

    R = 6372.8  # 3959.87433  this is in miles.  For Earth radius in kilometers use 6372.8 km
    if isinstance(lat1, collections.abc.Sequence):
        dLat = np.array([radians(l2 - l1) for l2, l1 in zip(lat2, lat1)])
        dLon = np.array([radians(l2 - l1) for l2, l1 in zip(lon2, lon1)])
        lat1 = np.array([radians(l1) for l1 in lat1])
        lat2 = np.array([radians(l2) for l2 in lat2])
    else:
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = R * c
    return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist


# calc. avg. distance to restaurants
def avg_Hdist_to_restaurants(courier_lat, courier_lon, restaurants_ids):
    return np.mean(
        [
            calc_haversine_dist(v["lat"], v["lon"], courier_lat, courier_lon)
            for v in restaurants_ids.values()
        ]
    )


def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=["category", "object"]))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except ValueError:
            print("Error encoding " + feature)
    return df


def feature_engineering(df: pd.DataFrame, dict_ids: Dict[str, Dict[str, int]]):
    df["dist_to_restaurant"] = calc_dist(
        df.courier_lat,
        df.courier_lon,
        df.restaurant_lat,
        df.restaurant_lon,
    )
    df["avg_dist_to_restaurants"] = [
        avg_dist_to_restaurants(lat, lon, dict_ids)
        for lat, lon in zip(df.courier_lat, df.courier_lon)
    ]
    df["Hdist_to_restaurant"] = calc_haversine_dist(
        df.courier_lat.tolist(),
        df.courier_lon.tolist(),
        df.restaurant_lat.tolist(),
        df.restaurant_lon.tolist(),
    )
    df["avg_Hdist_to_restaurants"] = [
        avg_Hdist_to_restaurants(lat, lon, dict_ids)
        for lat, lon in zip(df.courier_lat, df.courier_lon)
    ]

    return df
