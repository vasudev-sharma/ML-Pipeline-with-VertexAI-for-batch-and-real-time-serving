import logging
import numpy as np
import os
from trainer.training import save_model
import pandas as pd

from trainer.data_prep import get_restaurants_df, read_data
from trainer.features_processing import feature_engineering, Encoder
from trainer.clustering import run_clustering, order_busyness
from trainer.training import generate_ds, load_csv_data
from trainer.utils import get_config_file, upload_to_gcs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # SET THE LEVEL OF Logging in Python

# Deterministic imports
np.random.seed(1)


job_id = os.getenv("CLOUD_ML_JOB_ID")


if __name__ == "__main__":
    logging.info(f"The Job ID is: {job_id}")
    train_config = get_config_file("configs/training_config.yaml")

    flag_cloud_storage = False
    # Data processing
    # DO something with process data
    filename = train_config["data"]["filename_uri"]
    # print("Earlier filename is", filename)
    # print("type of data is", type(filename))
    if str(filename) == "None":  # TODO: fixme
        # print("local storage")
        filename = train_config["data"]["filename"]
        processed_df = read_data(filename=filename)  # TODO: READ
    else:
        processed_df = load_csv_data(
            bucket_name=train_config["data"]["bucket_name"],
            blob_path=train_config["data"]["blob_path"],
        )
        flag_cloud_storage = True

    print("The filename is ", filename)

    logging.info("***" * 10)
    logging.info(processed_df.head())

    logging.info("***" * 10)
    restaurants_df, restaurants_ids = get_restaurants_df(processed_df)

    # feature engineering
    logging.info("\n\n*********" * 10)
    logging.info(f"DataFrame after feature processing is: {restaurants_df}")
    restaurants_df = feature_engineering(restaurants_df, restaurants_ids)

    # clustering

    k = train_config["clustering"]["k"]
    resolution = train_config["clustering"]["resolution"]
    h3_clustered_df = run_clustering(k, restaurants_df, restaurants_ids, resolution)

    logging.info(h3_clustered_df.head())

    # Order busyness
    busyness_df = order_busyness(h3_clustered_df)
    logging.info(busyness_df.head())

    # TODO: Save cleaned up dataset

    # Label Encoding
    busyness_df["h3_index"] = busyness_df.h3_index.astype("category")
    busyness_df = Encoder(busyness_df)
    logging.info(busyness_df.head())

    # Training: data prep + training model
    # X_train, X_test, y_train, y_test = generate_ds(busyness_df, split_size=0.33, random_state=42)

    if flag_cloud_storage:
        # Save to GCP
        busyness_df.to_csv(
            f'{train_config["data"]["processed_filename_uri"]}_{job_id}/processed_data.csv',
            index=False,
        )
        logging.info(
            f"Saving processed data @: {train_config['data']['processed_filename_uri']}"
        )
    X, y = generate_ds(busyness_df)

    # Train + val, test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_config["data"]["test_size"],
        random_state=train_config["data"]["random_state"],
    )

    # Training, Valdation dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=train_config["data"]["val_size"],
        random_state=train_config["data"]["random_state"],
    )

    if flag_cloud_storage:

        # Training Data
        train_data_output = pd.concat((X_train, y_train), axis=1).to_csv(
            f'{train_config["data"]["processed_filename_uri"]}_{job_id}/train_dataset.csv',
            index=False,
        )
        # Validation
        val_data_output = pd.concat((X_val, y_val), axis=1).to_csv(
            f'{train_config["data"]["processed_filename_uri"]}_{job_id}/val_dataset.csv',
            index=False,
        )
        # Testing Data
        test_data_output = pd.concat((X_test, y_test), axis=1).to_csv(
            f'{train_config["data"]["processed_filename_uri"]}_{job_id}/test_dataset.csv',
            index=False,
        )

    # X_train, X_val, y_train, y_val = generate_ds(X_train, y_train,  split_size=0.33, random_state=42)

    regr = RandomForestRegressor(max_depth=4, random_state=0, n_jobs=-1)

    # Fitting the model
    regr.fit(X_train, y_train)
    regr.score(X_test, y_test)

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=regr,
        param_grid=train_config["model"]["grid_search"]["params"],
        cv=train_config["model"]["grid_search"]["cv"],
        n_jobs=train_config["model"]["grid_search"]["n_jobs"],
        verbose=1,
        scoring=train_config["model"]["grid_search"]["scoring"],
    )

    # Fit the Grid Search
    grid_search.fit(X_train, y_train)

    logging.info(grid_search.best_score_)

    # Get the best model
    rf_best = grid_search.best_estimator_
    rf_best

    model = rf_best

    # Environment variable for Vertex AI
    MODEL_DIR = os.getenv("AIP_MODEL_DIR")
    model_filepath = "model.pkl"
    if not MODEL_DIR:
        MODEL_DIR = ""  # Save it locally in model directly
        save_model(model, "model.pkl")
        logging.info("Model is saved as: model.pkl'")
    else:
        # Save the best model
        save_model(model, model_filepath)

        gcs_path = upload_to_gcs(
            bucket_name=train_config["model"]["bucket_name"],
            source_file_path=model_filepath,
            destination_blob_name=train_config["model"]["blob_path"],
        )
        logging.info(f"Model is saved to : {gcs_path}")
