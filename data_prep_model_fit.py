from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


def data_transform(red_wine_dataset: pd.DataFrame, white_wine_dataset: pd.DataFrame) -> pd.DataFrame :
    red_wine_dataset['wine_type'] = "red"
    white_wine_dataset['wine_type'] = "white"
    wine_dataset = pd.concat([red_wine_dataset, white_wine_dataset], axis=0)
    wine_dataset["wine_type"] = np.where(wine_dataset["wine_type"] == "red", 1, 0)
    return wine_dataset


def data_labels_change(wine_dataset: pd.DataFrame) -> tuple[Any, Any]:
    conditions = [(wine_dataset["quality"] <= 3),
        (wine_dataset.quality > 3) & (wine_dataset.quality <= 6),
        (wine_dataset["quality"] > 6)]
    values = [0, 1, 2]
    wine_dataset['quality'] = np.select(conditions, values)
    wine_dataset_train, wine_dataset_test = train_test_split(wine_dataset, test_size=0.4, shuffle=True)
    return wine_dataset_train, wine_dataset_test

def data_split(wine_dataset_train: pd.DataFrame, wine_dataset_test: pd.DataFrame) -> tuple[Any, Any, Any, Any]:
    X_train = wine_dataset_train.loc[:, wine_dataset_train.columns != "quality"]
    y_train = wine_dataset_train["quality"]
    X_test = wine_dataset_test.loc[:, wine_dataset_test.columns != "quality"]
    y_test = wine_dataset_test["quality"]
    return X_train, y_train, X_test, y_test

def lgbm_fit(X_train: pd.DataFrame, y_train: pd.DataFrame):
    categorical_features_names = ["wine_type"]
    features_names = X_train.columns
    cat_features_index = [index for index, feature_name in enumerate(features_names) if
                          feature_name in categorical_features_names]
    model_params = {
        'learning_rate': 0.1,
        'max_depth': None,
        'n_estimators': 500,
        'min_child_samples': 10,
        'categorical_feature': cat_features_index,
        'n_jobs': 1,
        'random_state': 1234,
    }
    lgbm_model = LGBMClassifier(**model_params)
    lgbm_model.fit(X_train, y_train)
    return lgbm_model, model_params
