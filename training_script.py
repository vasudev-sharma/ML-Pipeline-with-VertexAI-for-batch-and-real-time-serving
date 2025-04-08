import logging
from io import StringIO
import numpy as np
from training import save_model

from data_prep import get_restaurants_df, process_data
from features_processing import (
    feature_engineering,
    Encoder
)
from clustering import run_clustering, order_busyness
from training import generate_ds
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from utils import get_config_file



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # SET THE LEVEL OF Logging in Python

# Deterministic imports
np.random.seed(1)

if __name__ == "__main__":

    train_config = get_config_file('configs/training_config.yaml')


    # Data processing
    # DO something with process data
    filename = train_config['data']['filename']  # TODO: Add config.yaml for hardcoded values

    processed_df = process_data(filename=filename) # TODO: READ

    logging.debug("***" * 10)
    logging.debug(processed_df.head())

    logging.debug("***" * 10)
    restaurants_df, restaurants_ids = get_restaurants_df(processed_df)

    # Fetch couriers
    logging.debug(f"Length of unique elements is: {len(restaurants_ids)}")
    logging.debug(f"The restaurant dataframe is: {restaurants_df}")

    # feature engineering
    logging.debug("\n\n*********" * 10)
    logging.debug(f"DataFrame after feature processing is: {restaurants_df}")
    restaurants_df = feature_engineering(restaurants_df, restaurants_ids)
    logging.debug(type(restaurants_ids))
    logging.debug(restaurants_ids)

    # clustering

    k = train_config['clustering']['k']
    resolution = train_config['clustering']['resolution']
    h3_clustered_df = run_clustering(k, restaurants_df, restaurants_ids, resolution)

    logging.debug(h3_clustered_df.head())

    # Order busyness
    busyness_df = order_busyness(h3_clustered_df)
    logging.debug(busyness_df.head())



    # TODO: Save cleaned up dataset

    # Label Encoding
    busyness_df["h3_index"] = busyness_df.h3_index.astype("category")
    busyness_df= Encoder(busyness_df)
    logging.debug(busyness_df.head())

    # Training: data prep + training model
    # X_train, X_test, y_train, y_test = generate_ds(busyness_df, split_size=0.33, random_state=42)
    
    X, y = generate_ds(busyness_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_config['data']['test_size'], random_state=train_config['data']['random_state']
    )

    # X_train, X_val, y_train, y_val = generate_ds(X_train, y_train,  split_size=0.33, random_state=42)

    regr = RandomForestRegressor(max_depth=4, random_state=0, n_jobs=-1)

    # Fitting the model
    regr.fit(X_train, y_train)
    regr.score(X_test, y_test)




    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=regr, param_grid=train_config['model']['grid_search']['params'], cv=3, n_jobs=-1, verbose=1, scoring="r2"
    )

    # Fit the Grid Search
    grid_search.fit(X_train, y_train)

    logging.info(grid_search.best_score_)

    
    # Get the best model
    rf_best = grid_search.best_estimator_
    rf_best

    model = rf_best

    # Save the best model
    save_model(model, 'models/model.pkl')
    # TODO: Save to cloud storage instead 




