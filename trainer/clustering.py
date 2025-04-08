# clustering.py
import numpy as np
import pandas as pd # type: ignore
from features_processing import calc_dist
from typing import Dict
import h3


# STEP 1 - define K & initiate data
def initiate_centroids(k, df):
    """
    Select k data points as centroids
    k: number of centroids
    dset: pandas dataframe
    """
    centroids = df.sample(k)
    return centroids


# STEP 2 - define distance metric : Euclidean distance
def eucl_dist(p1x, p1y, p2x, p2y):
    """Finds the ec

    Args:
        p1x (_type_): _description_
        p1y (_type_): _description_
        p2x (_type_): _description_
        p2y (_type_): _description_

    Returns:
        _type_: _description_
    """
    return calc_dist(p1x, p1y, p2x, p2y)


# STEP 3 - Centroid assignment
def centroid_assignation(df, centroids):

    assignation = []

    assign_errors = []
    centroids_list = [c for i, c in centroids.iterrows()]
    for _, obs in df.iterrows():
        # Estimate error
        all_errors = [
            eucl_dist(
                centroid["lat"], centroid["lon"], obs["courier_lat"], obs["courier_lon"]
            )
            for centroid in centroids_list
        ]

        # Get the nearest centroid and the error
        nearest_centroid = np.where(all_errors == np.min(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.min(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)
    df["Five_Clusters_embedding"] = assignation
    df["Five_Clusters_embedding_error"] = assign_errors
    return df


def h3_clustering(resolution, df):

    # Issue with 
    df["courier_location_timestamp"] = pd.to_datetime(df["courier_location_timestamp"], format='mixed')
    df["order_created_timestamp"] = pd.to_datetime(df["order_created_timestamp"])
    df["h3_index"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for (lat, lon) in zip(df.courier_lat, df.courier_lon)
    ]  #
    df["date_day_number"] = [d for d in df.courier_location_timestamp.dt.day_of_year]
    df["date_hour_number"] = [d for d in df.courier_location_timestamp.dt.hour]

    return df


def run_clustering(
    k: int, df: pd.DataFrame, dict_ids: Dict[str, Dict[str, int]], resolution: int
):
    centroids_init = pd.DataFrame(
        [{"lat": v["lat"], "lon": v["lon"]} for v in dict_ids.values()]
    )
    centroids = initiate_centroids(k, centroids_init)

    # DataFrame of couriers
    df_couriers = pd.DataFrame({})
    df_couriers["lat"] = df["courier_lat"]
    df_couriers["lon"] = df["courier_lon"]

    # TODO: fix me
    df_centroids = centroid_assignation(df, centroids)

    h3_clustered_df = h3_clustering(resolution, df_centroids)

    return h3_clustered_df


def order_busyness(df: pd.DataFrame) -> pd.DataFrame:
    index_list = [
        (i, d, hr)
        for (i, d, hr) in zip(df.h3_index, df.date_day_number, df.date_hour_number)
    ]

    set_indexes = list(set(index_list))
    dict_indexes = {label: index_list.count(label) for label in set_indexes}
    df["orders_busyness_by_h3_hour"] = [dict_indexes[i] for i in index_list]

    restaurants_counts_per_h3_index = {
        a: len(b)
        for a, b in zip(
            df.groupby("h3_index")["restaurant_id"].unique().index,
            df.groupby("h3_index")["restaurant_id"].unique(),
        )
    }
    df["restaurants_per_index"] = [
        restaurants_counts_per_h3_index[h] for h in df.h3_index
    ]

    return df


# if __name__ == "__main__":

#     df_restaurants = pd.DataFrame(
#         [{"lat": v["lat"], "lon": v["lon"]} for v in restaurants_ids.values()]
#     )
#     centroids = initiate_centroids(k, df_restaurants)

#     # DataFrame of couriers
#     df_couriers = pd.DataFrame({})
#     df_couriers["lat"] = restaurants_df["courier_lat"]
#     df_couriers["lon"] = restaurants_df["courier_lon"]

#     # TODO: fix me
#     df_centroids = centroid_assignation(restaurants_df, centroids)

#     resolution = 7
#     h3_clustered_df = h3_clustering(resolution, restaurants_df)
