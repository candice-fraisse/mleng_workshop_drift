from typing import Optional

import pandas as pd
from deepchecks import Dataset, checks
from deepchecks.tabular.checks import TrainTestFeatureDrift, TrainTestPredictionDrift
from sklearn.ensemble import RandomForestClassifier

from skidless import drift


def deepcheck_result_feature_drift(result, val):
    """
    Prints an alarm or warning if the value of a feature drift is higher than a threshold
    """
    for i in val:
        if 0.2 > result.value[i]['Drift score'] >= 0.1:
            print("Warning", i)
        elif result.value[i]['Drift score'] >= 0.2:
            print("Alarm", i)


def deepcheck_result_prediction_drift(result):
    """
       Prints an alarm or warning if the value of the prediction drift is higher than a threshold
       """
    if 0.2 > result.value['Drift score'] >= 0.1:
        print("Warning")
    elif result.value['Drift score'] >= 0.2:
        print("Alarm")
    else:
        print('No drift detected')


def deepcheck_dataset_drift(result):
    """
    Prints an alarm or warning if the domain classifier drift score is higher than a threshold
    """
    if 0.2 > result.value['domain_classifier_drift_score'] >= 0.1:
        print("Warning")
    elif result.value['domain_classifier_drift_score'] >= 0.2:
        print("Alarm")
    else:
        print('No drift detected')


def deepcheck_detect_drift(data_train: pd.DataFrame, data_to_compare: pd.DataFrame,
                           label_col: str, cat_features: list,
                           model: Optional = RandomForestClassifier(max_depth=5, n_jobs=-1),
                           test_type: str = "feature_drift"):
    """
    Prints an alarm or a warning if a feature drift has happened
    """
    train_drifted_ds = Dataset(data_train, label=label_col, cat_features=cat_features)
    test_drifted_ds = Dataset(data_to_compare, label=label_col, cat_features=cat_features)
    if test_type == "feature_drift":
        check = TrainTestFeatureDrift()
        result = check.run(train_dataset=train_drifted_ds, test_dataset=test_drifted_ds)
        val = data_to_compare.loc[:, data_to_compare.columns != label_col].columns
        deepcheck_result_feature_drift(result, val)
    elif test_type == "prediction_drift":
        check = TrainTestPredictionDrift()
        model = model.fit(train_drifted_ds.data[train_drifted_ds.features],
                          train_drifted_ds.data[train_drifted_ds.label_name])
        result = check.run(train_dataset=train_drifted_ds, test_dataset=test_drifted_ds, model=model)
        deepcheck_result_prediction_drift(result)
    elif test_type == "dataset_drift":
        check = checks.WholeDatasetDrift(verbose=False)
        result = check.run(train_dataset=train_drifted_ds, test_dataset=test_drifted_ds)
        deepcheck_dataset_drift(result)
    else:
        print("Not recognised")


def deepcheck_detect_gradual_drift(data_train: pd.DataFrame,
                                   data_to_compare: pd.DataFrame,
                                   column_name: str,
                                   label_col: str,
                                   cat_features: list,
                                   value_drift: float,
                                   model,
                                   action: str = "increase",
                                   test_type: str = "feature_drift",
                                   nb_sample: int = 100,
                                   nb_days: int = 5):
    """
    Generates a gradual drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    days = 0
    init_value_drift = value_drift
    data_generated = drift.dataset_generator_yield(data=data_to_compare, nb_sample=nb_sample)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_generator(data_generated=dataset_sample,
                                                  column_name=column_name,
                                                  value_of_drift=value_drift,
                                                  action=action)
        deepcheck_detect_drift(data_train,
                               dataset_corrupted,
                               label_col=label_col,
                               cat_features=cat_features,
                               model=model,
                               test_type=test_type)
        days += 1
        value_drift += init_value_drift


def deepcheck_detect_seasonal_drift(data_train: pd.DataFrame,
                                    data_to_compare: pd.DataFrame,
                                    column_name: str,
                                    label_col: str,
                                    value_drift: float,
                                    frequency: int,
                                    cat_features: list,
                                    model,
                                    action: str = "increase",
                                    test_type: str = "feature_drift",
                                    nb_sample: int = 100,
                                    nb_days: int = 5):
    """
    Generates a seasonal drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    days = 0
    init_value_drift = value_drift
    seasonal_days = drift.generate_frequency(nb_day=nb_days, frequency=frequency)
    data_generated = drift.dataset_generator_yield(data=data_to_compare, nb_sample=nb_sample)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_seasonal_days_only(seasonal_days=seasonal_days,
                                                           days=days,
                                                           data_generated=dataset_sample,
                                                           column_name=column_name,
                                                           value_of_drift=value_drift,
                                                           action=action)
        deepcheck_detect_drift(data_train,
                               dataset_corrupted,
                               label_col=label_col,
                               cat_features=cat_features,
                               model=model,
                               test_type=test_type)
        days += 1
        value_drift += init_value_drift
