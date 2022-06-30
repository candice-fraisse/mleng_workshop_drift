import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skmultiflow.drift_detection import EDDM, ADWIN, HDDM_W

from skidless import drift


def drift_eddm(error_rate, list_experiment_results):
    """
    Detects if a drift has happened using EDDM detector
    """
    eddm = EDDM()
    index_alarm = []
    for i in range(len(error_rate)):
        eddm.add_element(error_rate[i])
        if eddm.detected_warning_zone():
            list_experiment_results.append(True)
            index_alarm.append(i)
        if eddm.detected_change():
            list_experiment_results.append(True)
            index_alarm.append(i)
    return list_experiment_results, index_alarm


def drift_hddm_w(error_rate, list_experiment_results):
    """
    Detects if a drift has happened using HDDM_W detector
    """
    hddm_w = HDDM_W()
    index_alarm = []
    for i in range(len(error_rate)):
        hddm_w.add_element(error_rate[i])
        if hddm_w.detected_warning_zone():
            list_experiment_results.append(True)
            index_alarm.append(i)
        if hddm_w.detected_change():
            list_experiment_results.append(True)
            index_alarm.append(i)
    return list_experiment_results, index_alarm


def drift_adwin(error_rate: np.array, list_experiment_results: list):
    """
    Detects if a drift has happened using ADWIN detector
    """
    error_rate_float = error_rate.astype('float32')
    index_alarm = []
    adwin = ADWIN()
    for i in range(len(error_rate_float)):
        adwin.add_element(error_rate_float[i])
        if adwin.detected_warning_zone():
            list_experiment_results.append(True)
            index_alarm.append(i)
        if adwin.detected_change():
            list_experiment_results.append(True)
            index_alarm.append(i)
    return list_experiment_results, index_alarm


def drift_with_labels(error_rate: np.array, method: str = "HDDM_W"):
    """
    Depending on the chosen detector, a given function will be chosen
    """
    list_experiment_results = []
    if method == "EDDM":
        list_experiment_results, index_alarm = drift_eddm(error_rate, list_experiment_results)
    elif method == "ADWIN":
        list_experiment_results, index_alarm = drift_adwin(error_rate, list_experiment_results)
    elif method == "HDDM_W":
        list_experiment_results, index_alarm = drift_hddm_w(error_rate, list_experiment_results)
    else:
        list_experiment_results, index_alarm = None, None
    return list_experiment_results, index_alarm


def drift_detector_with_labels_test(data_to_compare: pd.DataFrame,
                                    label_col: str,
                                    model,
                                    test_name: str = "HDDM_W"):
    """
    Compare each variable of the two dataframes with a chosen labeled test
    """
    plt.figure(figsize=(20, 4))
    x_to_compare_sample = data_to_compare.loc[:, data_to_compare.columns != label_col]
    y_to_compare_sample = np.array(data_to_compare[label_col])
    y_hat = np.array(model.predict(x_to_compare_sample))
    if test_name == "ADWIN":
        error_rate = (y_to_compare_sample == y_hat).astype(int)
    else:
        error_rate = (y_to_compare_sample != y_hat).astype(int)
    error_rate = np.array(error_rate)
    has_data_drifted, index_alarm = drift_with_labels(error_rate, method=test_name)
    plt.plot(error_rate)
    plt.show()
    return has_data_drifted, index_alarm


def drift_detector_labels_gradual_drift(data_train: pd.DataFrame,
                                        data_to_compare: pd.DataFrame,
                                        column_name: str,
                                        label_col: str,
                                        model,
                                        value_drift: float,
                                        action: str = "increase",
                                        test_name: str = "HDDM_W",
                                        nb_sample: int = 100,
                                        nb_days: int = 5):
    """
    Generates a gradual drift
    Prints an alarm or a warning if a feature drift or anomaly has happened throughout a period of days
    """
    days = 0
    init_value_drift = value_drift
    data_generated = drift.dataset_generator_yield(data=data_to_compare, nb_sample=nb_sample)
    data_reference_sample = data_train.sample(n=nb_sample)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_generator(data_generated=dataset_sample,
                                                  column_name=column_name,
                                                  value_of_drift=value_drift,
                                                  action=action)
        full_data = pd.concat([data_reference_sample, dataset_corrupted], axis=0)
        has_data_drifted, index_alarm = drift_detector_with_labels_test(data_to_compare=full_data,
                                                                        label_col=label_col,
                                                                        model=model,
                                                                        test_name=test_name)
        print(has_data_drifted)
        days += 1
        value_drift += init_value_drift


def drift_detector_labels_seasonal_drift(data_train: pd.DataFrame,
                                         data_to_compare: pd.DataFrame,
                                         column_name: str,
                                         label_col: str,
                                         model,
                                         value_drift: float,
                                         frequency: int,
                                         test_name: str = "HDDM_W",
                                         action: str = "increase",
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
    data_reference_sample = data_train.sample(n=nb_sample)
    while days <= nb_days:
        print("Day", days, ":")
        dataset_sample = next(data_generated)
        dataset_corrupted = drift.drift_seasonal_days_only(seasonal_days=seasonal_days,
                                                           days=days,
                                                           data_generated=dataset_sample,
                                                           column_name=column_name,
                                                           value_of_drift=value_drift,
                                                           action=action)
        full_data = pd.concat([data_reference_sample, dataset_corrupted], axis=0)
        has_data_drifted = drift_detector_with_labels_test(data_to_compare=full_data,
                                                           label_col=label_col,
                                                           model=model,
                                                           test_name=test_name)
        print(has_data_drifted)
        days += 1
        value_drift += init_value_drift
