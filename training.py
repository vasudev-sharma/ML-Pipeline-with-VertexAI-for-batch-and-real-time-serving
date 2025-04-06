import logging
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


def Encoder(df: pd.DataFrame) -> pd.DataFrame:
    columnsToEncode = list(df.select_dtypes(include=["category", "object"]))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            logging.info("Error encoding " + feature)
    return df



def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_train, X_test, y_train, y_test


# if __name__ == "__main__":
#     X = df[
#         [
#             "dist_to_restaurant",
#             "Hdist_to_restaurant",
#             "avg_Hdist_to_restaurants",
#             "date_day_number",
#             "restaurant_id",
#             "Five_Clusters_embedding",
#             "h3_index",
#             "date_hour_number",
#             "restaurants_per_index",
#         ]
#     ]
#     y = df[["orders_busyness_by_h3_hour"]]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42
#     )

    regr = RandomForestRegressor(max_depth=4, random_state=0, n_jobs=-1)

    # Fitting the model
    regr.fit(X_train, y_train)
    regr.score(X_test, y_test)

    # TODO: Mange parameters: (Maybe Hydra)
    params = {
        "max_depth": [4, 5],
        "min_samples_leaf": [50, 75],
        "n_estimators": [100, 150],
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=regr, param_grid=params, cv=3, n_jobs=-1, verbose=1, scoring="r2"
    )

    # Fit the Grid Search
    grid_search.fit(X_train, y_train)

    grid_search.best_score_
