import warnings
import json
import pandas as pd
from evidently import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, \
    NumTargetDriftProfileSection

import drift

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def evidently_ai_detect_drift(data_train: pd.DataFrame,
                              data_to_compare: pd.DataFrame,
                              cat_features: list,
                              label_col: str):
    """
    Prints an alarm or a warning if a feature or label drift has happened
    """
    target = label_col
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.categorical_features = cat_features
    data_and_target_drift_profile = Profile(sections=[DataDriftProfileSection(), NumTargetDriftProfileSection()])
    data_and_target_drift_profile.calculate(data_train, data_to_compare, column_mapping=column_mapping)
    results = data_and_target_drift_profile.json()
    results = json.loads(results)
    val = data_to_compare.columns
    for i in val:
        if results['data_drift']["data"]["metrics"][i]['drift_detected'] == True:
            print('Alarm', i)


def evidently_ai_detect_gradual_drift(data_train: pd.DataFrame,
                                      data_to_compare: pd.DataFrame,
                                      column_name: str,
                                      label_col: str,
                                      cat_features: list,
                                      value_drift: float,
                                      action: str = "increase",
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
        evidently_ai_detect_drift(data_train=data_train,
                                  data_to_compare=dataset_corrupted,
                                  cat_features=cat_features,
                                  label_col=label_col)
        days += 1
        value_drift += init_value_drift


def evidently_ai_detect_seasonal_drift(data_train: pd.DataFrame,
                                       data_to_compare: pd.DataFrame,
                                       column_name: str,
                                       label_col: str,
                                       cat_features: list,
                                       value_drift: float,
                                       frequency: int,
                                       action: str = "increase",
                                       nb_sample: int = 100,
                                       nb_days: int = 5):
    """
    Generates a seasonal drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    days = 0
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
        evidently_ai_detect_drift(data_train=data_train,
                                  data_to_compare=dataset_corrupted,
                                  cat_features=cat_features,
                                  label_col=label_col)
        days += 1
