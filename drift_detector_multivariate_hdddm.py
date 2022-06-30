import pandas as pd
from menelaus.data_drift import hdddm

from skidless import drift

# github.com/mitre/menelaus/blob/dev/src/menelaus/data_drift/histogram_density_method.py

def hdddm_detect_drift(data_train: pd.DataFrame,
                       data_to_compare: pd.DataFrame,
                       gamma_level: float = 0.05):
    """
    Prints an alarm or a warning if a feature or label drift has happened
    """
    HDM = hdddm.HistogramDensityMethod(divergence="H", detect_batch=1, statistic="stdev",
                                       significance=gamma_level, subsets=4)
    detector = hdddm.HDDDM(HDM)
    detector.set_reference(data_train)
    detector.update(data_to_compare)
    print(detector.drift_state)

def hdddm_detect_gradual_drift(data_train: pd.DataFrame,
                               data_to_compare: pd.DataFrame,
                               column_name: str,
                               value_drift: float,
                               nb_sample: int = 100,
                               nb_days: int = 5,
                               action: str = "increase",
                               gamma_level: float = 0.05):
    """
    Generates a gradual drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    days = 0
    init_value_drift = 5
    data_generated = drift.dataset_generator_yield(data=data_to_compare, nb_sample=nb_sample)
    HDM = hdddm.HistogramDensityMethod(divergence="H", detect_batch=1, statistic="stdev",
                                       significance=gamma_level, subsets=4)
    detector = hdddm.HDDDM(HDM)
    detector.set_reference(data_train)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_generator(data_generated=dataset_sample,
                                                  column_name=column_name,
                                                  value_of_drift=value_drift,
                                                  action=action)
        detector.update(dataset_corrupted)
        print(detector.drift_state)
        days += 1
        value_drift += init_value_drift


def hdddm_detect_seasonal_drift(data_train: pd.DataFrame,
                                data_to_compare: pd.DataFrame,
                                column_name: str,
                                value_drift: float,
                                frequency: int,
                                nb_sample: int = 100,
                                nb_days: int = 5,
                                action: str = "increase",
                                gamma_level: float = 0.05):
    """
    Generates a seasonal drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    days = 0
    init_value_drift = value_drift
    seasonal_days = drift.generate_frequency(nb_day=nb_days, frequency=frequency)
    data_generated = drift.dataset_generator_yield(data=data_to_compare, nb_sample=nb_sample)
    HDM = hdddm.HistogramDensityMethod(divergence="H", detect_batch=1, statistic="stdev",
                                       significance=gamma_level, subsets=4)
    detector = hdddm.HDDDM(HDM)
    detector.set_reference(data_train)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_seasonal_days_only(seasonal_days=seasonal_days,
                                                           days=days,
                                                           data_generated=dataset_sample,
                                                           column_name=column_name,
                                                           value_of_drift=value_drift,
                                                           action=action)
        detector.update(dataset_corrupted)
        print(detector.drift_state)
        days += 1
        value_drift += init_value_drift
