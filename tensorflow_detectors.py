import warnings
import pandas as pd
import tensorflow_data_validation as tfdv
import drift

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def tensorflow_generate_format(dataset: pd.DataFrame):
    """
    Generates the tensorflow format for a dataset
    """
    dataset.to_csv('dataset.csv')
    dataset_stats = tfdv.generate_statistics_from_csv(data_location="dataset.csv")
    return dataset_stats


def tensorflow_prepare_detection(data_train: pd.DataFrame):
    """
    Returns the training schema and tensorflow format for train dataset
    """
    dataset_stats = tensorflow_generate_format(data_train)
    schema = tfdv.infer_schema(dataset_stats)
    return schema, dataset_stats


def tensorflow_detect_drift(data_train: pd.DataFrame,
                            data_to_compare: pd.DataFrame,
                            label_col: str,
                            alpha_level: float = 0.01):
    """
    Prints an alarm or a warning if a feature drift or anomaly has happened
    """
    X_train = data_train.loc[:,data_train.columns != label_col]
    X_test_corrupted = data_to_compare.loc[:,data_to_compare.columns != label_col]
    schema, dataset_stats = tensorflow_prepare_detection(X_train)
    serving_stats = tensorflow_generate_format(X_test_corrupted)
    for col in X_test_corrupted.columns:
        if X_test_corrupted[col].dtype == ["category", "string"]:
            tfdv.get_feature(schema, col).skew_comparator.infinity_norm.threshold = alpha_level
            skew_anomalies = tfdv.validate_statistics(
                statistics=dataset_stats, schema=schema, serving_statistics=serving_stats)
        elif X_test_corrupted[col].dtype != ["category", "string"]:
            tfdv.get_feature(schema, col).skew_comparator.jensen_shannon_divergence.threshold = alpha_level
            skew_anomalies = tfdv.validate_statistics(
                statistics=dataset_stats, schema=schema, serving_statistics=serving_stats)
    tfdv.display_anomalies(skew_anomalies)


def tensorflow_detect_gradual_drift(data_train: pd.DataFrame,
                                    data_to_compare: pd.DataFrame,
                                    label_col: str,
                                    column_name: str,
                                    value_drift: float,
                                    nb_sample: int = 100,
                                    nb_days: int = 5,
                                    alpha_level: float = 0.01):
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
        dataset_corrupted = drift.drift_generator_univariate_multiply(data=dataset_sample,
                                                                      column_name=column_name,
                                                                      value=value_drift)
        tensorflow_detect_drift(data_train, dataset_corrupted, label_col, alpha_level=alpha_level)
        days += 1
        value_drift += init_value_drift


def tensorflow_detect_seasonal_drift(data_train: pd.DataFrame,
                                     data_to_compare: pd.DataFrame,
                                     column_name: str,
                                     label_col: str,
                                     value_drift: float,
                                     frequency: int,
                                     action: str = "increase",
                                     nb_sample: int = 100,
                                     nb_days: int = 5,
                                     alpha_level: float = 0.01):
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
        tensorflow_detect_drift(data_train, dataset_corrupted, label_col, alpha_level=alpha_level)
        days += 1
