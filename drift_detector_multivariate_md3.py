import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.svm import LinearSVC

import drift


def fit_linear_SVC(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Fits a linear SVC model
    """
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model


def calc_marginal_density(model, X: pd.DataFrame, y: pd.DataFrame):
    """
    Calculates the margin density
    """
    w = model.coef_
    w = w.T
    b = model.intercept_
    dp = abs(np.dot(X, w) + b)
    rho = np.round(np.array(dp < 1).sum() / X.shape[0], 7)
    return rho


def calc_sigma_ref(model, X_train: pd.DataFrame, y_train: pd.DataFrame) -> float:
    """
    Initialises the reference sigma for the MD3 algorithm
    """
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=4369)
    cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=calc_marginal_density)
    sigma_ref = cv_results['test_score'].std()
    return sigma_ref


def initialise_md3(data_train: pd.DataFrame, label_col: str):
    X_train = data_train.loc[:, data_train.columns != label_col]
    y_train = data_train[label_col]
    model = fit_linear_SVC(X_train, y_train)
    rho_min = calc_marginal_density(model, X_train, y_train)
    sigma_ref = calc_sigma_ref(model, X_train, y_train)
    return model, rho_min, sigma_ref


def md3_detect_drift(data_train: pd.DataFrame,
                     data_to_compare: pd.DataFrame,
                     label_col: str):
    """
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    model, rho_min, sigma_ref = initialise_md3(data_train, label_col)
    X_corrupted = data_to_compare.loc[:, data_to_compare.columns != label_col]
    y_corrupted = data_to_compare[label_col]
    rho = calc_marginal_density(model, X_corrupted, y_corrupted)
    if abs(rho - rho_min) > (0.25 * sigma_ref):
        print("Alarm, drift Detected")


def md3_gradual_drift(data_train: pd.DataFrame,
                      data_to_compare: pd.DataFrame,
                      column_name: str,
                      label_col: str,
                      value_drift: float,
                      action: str = "increase",
                      nb_sample: int = 100,
                      nb_days: int = 5):
    """
    Generates a gradual drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    densities = []
    days = 0
    init_value_drift = value_drift
    model, rho_min, sigma_ref = initialise_md3(data_train, label_col)
    data_generated = drift.dataset_generator_yield(data=data_to_compare, nb_sample=nb_sample)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_generator(data_generated=dataset_sample,
                                                  column_name=column_name,
                                                  value_of_drift=value_drift,
                                                  action=action)
        X_corrupted = dataset_corrupted.loc[:, dataset_corrupted.columns != label_col]
        y_corrupted = dataset_corrupted[label_col]
        rho = calc_marginal_density(model, X_corrupted, y_corrupted)
        densities.append(rho)
        if abs(rho - rho_min) > (0.25 * sigma_ref):
            print("Alarm, drift Detected")
            rho_min = rho
        days += 1
        value_drift += init_value_drift


def md3_seasonal_drift(data_train: pd.DataFrame,
                       data_to_compare: pd.DataFrame,
                       column_name: str,
                       label_col: str,
                       value_drift: float,
                       frequency: int,
                       action: str = "increase",
                       nb_sample: int = 100,
                       nb_days: int = 5):
    """
    Generates a seasonal drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    densities = []
    days = 0
    model, rho_min, sigma_ref = initialise_md3(data_train, label_col)
    data_generated = drift.dataset_generator_yield(data=data_to_compare, nb_sample=nb_sample)
    seasonal_days = drift.generate_frequency(nb_day=nb_days, frequency=frequency)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_seasonal_days_only(seasonal_days=seasonal_days,
                                                           days=days,
                                                           data_generated=dataset_sample,
                                                           column_name=column_name,
                                                           value_of_drift=value_drift,
                                                           action=action)
        X_corrupted = dataset_corrupted.loc[:, dataset_corrupted.columns != label_col]
        y_corrupted = dataset_corrupted[label_col]
        rho = calc_marginal_density(model, X_corrupted, y_corrupted)
        densities.append(rho)
        if abs(rho - rho_min) > (0.25 * sigma_ref):
            print("Alarm, drift Detected")
            rho_min = rho
        days += 1
