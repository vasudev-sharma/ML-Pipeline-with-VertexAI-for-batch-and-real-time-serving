import logging
from io import StringIO
from typing import Tuple

import numpy as np
import pandas as pd


def process_data(filename: str, sample_data: StringIO) -> pd.DataFrame:
    # TODO: Add docstrings and Type hints
    try:
        logging.info(f"Reading from filename: {filename}")
        df = pd.read_csv(filename)
    except:
        logging.info("Reading from sample data")
        df = pd.read_csv(sample_data)

    df.dropna(axis=0, inplace=True)
    print(df.head())
    return df


def get_unique_restauraunts(processed_df) -> Tuple[pd.DataFrame, int]:
    # TODO: Add type hints and docstrings

    # unique restaurants
    restaurants_ids = {}
    list_restaurants_ids = []
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

    # Unique restaurants
    restaurants_df, num_restauraunts_ids = get_unique_restauraunts(df)
    return restaurants_df, num_restauraunts_ids


# if __name__ == "__main__":
#   SAMPLE_DATA = StringIO("""courier_id,order_number,courier_location_timestamp,courier_lat,courier_lon,order_created_timestamp,restaurant_lat,restaurant_lon
#                   a98737cbhoho5012hoho4b5bhoho867fhoho8475c658546d,281289453,2021-04-02T04:30:42.328Z,50.484520325268576,-104.61887559561548,2021-04-02T04:20:42Z,50.483696253259296,-104.61434958504181
#                   39a26fa0hohof428hoho47a4hohoa320hoho12e3d831c23a,280949566,2021-04-01T06:14:47.386Z,50.44257272227587,-104.55046328384749,2021-04-01T06:05:18Z,50.4424223,-104.5504874
#                   3813235ehoho7a42hoho4601hohob7eahoho799e8af5b535,281328578,2021-04-02T05:48:57.224Z,50.4959203,-104.6356053,2021-04-02T05:13:26Z,50.4965949,-104.6356059
#                   9f033953hohocd53hoho488ahohoaf51hohoc57943e499ed,281317998,2021-04-02T05:12:17.252Z,50.449445492444362,-104.61152056643822,2021-04-02T04:59:57Z,50.4495041,-104.6110744
#                   56f65bc8hohoba54hoho47dfhohoa09chohof7464b5d9848,281314132,2021-04-02T05:15:38.266Z,50.4952544,-104.6663829,2021-04-02T04:54:53Z,50.4951598,-104.6657333
#                   14995ac3hohoa1cbhoho4c4chohoab10hoho224f17643c1d,281338111,2021-04-02T05:56:05.205Z,50.495919191814679,-104.64014471092294,2021-04-02T05:25:31Z,50.496378,-104.6403723
#                   06c6be58hoho08e1hoho41a7hohob6f8hoho4dceb6c4fb80,281343348,2021-04-02T05:55:23.294Z,50.448327793713425,-104.53646034046066,2021-04-02T05:32:04Z,50.4482338434,-104.536186627
#                   4407d22bhoho005bhoho4ea1hohob2d9hoho4ad22c0dd75f,280950717,2021-04-01T06:19:07.133Z,50.472845499999991,-104.61854733333332,2021-04-01T06:07:15Z,50.4727424,-104.6185983
#                   53510a74hohoc305hoho47e9hoho95b2hoho401358e7fe9a,281313990,2021-04-02T05:15:18.303Z,50.44518463051854,-104.55607211858168,2021-04-02T04:54:41Z,50.4451323499,-104.556314434
#                   e75db857hoho17e0hoho4ddbhoho9884hoho4160f0f5f253,281248939,2021-04-02T03:30:31.258Z,50.452009655710675,-104.61147389130386,2021-04-02T03:19:09Z,50.451066433451821,-104.61137957374955
#                   """)
#   processed_df = process_data(filename='data.csv', sample_data=SAMPLE_DATA)


#   #Fetch couriers
#   len(processed_df.courier_id.unique())

#   # Unique restaurants
#   restaurants_df = get_unique_restauraunts(processed_df)
#   print(restaurants_df)
