from typing import Optional

import pandas as pd
import numpy as np
from numpy import random


def drift_generator_univariate_increase(data: pd.DataFrame, column_name: str, value: float) -> pd.DataFrame:
    """
    Increase variable named column_name by value
    Example:
        dataset_drifted = drift_generator_univariate_increase(data=input_dataset,
                                                              column_name="age",
                                                              value=1)
    """
    drifted_data = data.copy()
    drifted_data[column_name] += value
    return drifted_data


def drift_generator_univariate_decrease(data: pd.DataFrame, column_name: str, value: float) -> pd.DataFrame:
    """
    Decrease variable named column_name by value
    Example:
        dataset_drifted = drift_generator_univariate_decrease(data=input_dataset,
                                                              column_name="age",
                                                              value=1)
    """
    drifted_data = data.copy()
    drifted_data[column_name] -= value
    return drifted_data


def drift_generator_univariate_multiply(data: pd.DataFrame, column_name: str, value: float) -> pd.DataFrame:
    """
    Multiply variable named column_name by value
    Example:
        dataset_drifted = drift_generator_univariate_multiply(data=input_dataset,
                                                              column_name="age",
                                                              value=1)
    """
    drifted_data = data.copy()
    drifted_data[column_name] *= value
    return drifted_data


def drift_generator_univariate_divide(data: pd.DataFrame, column_name: str, value: float) -> pd.DataFrame:
    """
    Divide variable named column_name by value
    Example:
        dataset_drifted = drift_generator_univariate_divide(data=input_dataset,
                                                              column_name="age",
                                                              value=1)
    """
    drifted_data = data.copy()
    drifted_data[column_name] /= value
    return drifted_data


def drift_generator_univariate_change_to_normal(data: pd.DataFrame, column_name: str, seed: int = 202,
                                                mu: int = 0, sigma: int = 1) -> pd.DataFrame:
    """
    Change the variable distribution to a normal distribution with custom mu and sigma
    Example:
        dataset_drifted = drift_generator_univariate_change_to_normal(data=input_dataset,
                                                              column_name="age",
                                                              mu=0, sigma=2, seed=202)
    """
    drifted_data = data.copy()
    np.random.seed(seed)
    length = drifted_data.shape[0]
    normal_data = np.random.normal(mu, sigma, length)
    drifted_data[column_name] = normal_data
    return drifted_data


def drift_generator_univariate_change_to_poisson(data: pd.DataFrame, column_name: str, seed: int = 202,
                                                 lamb: int = 1) -> pd.DataFrame:
    """
    Change the variable distribution to a poisson distribution with custom lambda
    Example:
        dataset_drifted = drift_generator_univariate_change_to_poisson(data=input_dataset,
                                                              column_name="age",
                                                              lamb=1, seed=202)
    """
    drifted_data = data.copy()
    np.random.seed(seed)
    length = drifted_data.shape[0] + 1
    poisson_data = np.random.poisson(lamb, length)
    drifted_data[column_name] = poisson_data
    return drifted_data


def drift_generator_univariate_change_to_random(data: pd.DataFrame, column_name: str, seed: int = 202) -> pd.DataFrame:
    """
    Change the variable distribution to a random distribution
    Example:
        dataset_drifted = drift_generator_univariate_change_to_random(data=input_dataset,
                                                              column_name="age", seed=202)
    """
    drifted_data = data.copy()
    np.random.seed(seed)
    length = drifted_data.shape[0] + 1
    random_data = np.random.poisson(length)
    drifted_data[column_name] = random_data
    return drifted_data


def drift_generator_univariate_categorical_change(data: pd.DataFrame, column_name: str,
                                                  value1: str, value2: str) -> pd.DataFrame:
    """
   Swap the two categories in a column
   Example:
       dataset_drifted = drift_generator_univariate_categorical_change(data=input_dataset,
                                                             column_name="gender",value1= "male", value2="female")
   """
    drifted_data = data.copy()
    drifted_data[column_name] = drifted_data[column_name].astype("str")
    drifted_data.loc[drifted_data[column_name] == value1, column_name] = 'temp1'
    drifted_data.loc[drifted_data[column_name] == value2, column_name] = value1
    drifted_data.loc[drifted_data[column_name] == 'temp1', column_name] = value2
    return drifted_data


def drift_generator_concept_drift(data: pd.DataFrame, column_name: str, value: float,
                                  label_col: str, label_value: int, action: str) -> pd.DataFrame:
    """
    Induces a concept drift on the dataset
    Example:
        dataset_drifted = drift_generator_concept_drift(data=input_dataset,
                                                        column_name="age",
                                                        value=30,
                                                        label_col = "income_bracket",
                                                        label_value = 1,
                                                        action = "lower")
    """
    drifted_data = data.copy()
    if action == "greater":
        drifted_data.loc[drifted_data[column_name] > value, label_col] = label_value
    elif action == "lower":
        drifted_data.loc[drifted_data[column_name] < value, label_col] = label_value
    elif action == "equal":
        drifted_data.loc[drifted_data[column_name] == value, label_col] = label_value
    elif action == "different":
        drifted_data.loc[drifted_data[column_name] != value, label_col] = label_value
    return drifted_data


def dataset_generator_yield(data: pd.DataFrame, nb_sample: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generates a random sample drawn from a given dataset
    """
    while True:
        if seed:
            data_generated = data.sample(n=nb_sample, random_state=seed, axis=0, ignore_index=True)
            yield data_generated
        else:
            data_generated = data.sample(n=nb_sample, axis=0, ignore_index=True)
            yield data_generated


def generate_frequency(nb_day: int, frequency: int) -> list:
    """
    Generates a list of days according to a frequency
    """
    days = 0
    seasonal_days = []
    while days <= nb_day:
        seasonal_days.append(days)
        days += frequency
    return seasonal_days


def drift_generator(data_generated: pd.DataFrame, column_name: str,
                    value_of_drift: float = 1, value_categorical: list = ["val1", "val2"],
                    action: str = "increase") -> pd.DataFrame:
    """
    Generates a drift on a specific column with a specific action
    """
    if action == "increase":
        dataset_sample_corrupted = drift_generator_univariate_increase(data_generated, value=value_of_drift,
                                                                       column_name=column_name)
    elif action == "decrease":
        dataset_sample_corrupted = drift_generator_univariate_decrease(data_generated, value=value_of_drift,
                                                                       column_name=column_name)
    elif action == "divide":
        dataset_sample_corrupted = drift_generator_univariate_divide(data_generated, value=value_of_drift,
                                                                     column_name=column_name)
    elif action == "multiply":
        dataset_sample_corrupted = drift_generator_univariate_multiply(data_generated, value=value_of_drift,
                                                                       column_name=column_name)
    elif action == "categorical":
        dataset_sample_corrupted = drift_generator_univariate_categorical_change(data=data_generated,
                                                                                 value1=value_categorical[0],
                                                                                 value2=value_categorical[1],
                                                                                 column_name=column_name)
    return dataset_sample_corrupted


def drift_seasonal_days_only(seasonal_days: list,
                             days: int,
                             data_generated: pd.DataFrame,
                             column_name: str,
                             value_of_drift: Optional[float] = None,
                             value_categorical: Optional[list] = None,
                             action: str = "increase"):
    """
    Generates drifts on specific days only
    """
    seasonal_days = set(seasonal_days)
    if days in seasonal_days:
        return drift_generator(data_generated=data_generated,
                               column_name=column_name,
                               value_of_drift=value_of_drift,
                               value_categorical=value_categorical,
                               action=action)
    else:
        return data_generated


