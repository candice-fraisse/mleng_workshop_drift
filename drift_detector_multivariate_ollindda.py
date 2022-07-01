import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from typing import Dict, List

import drift


def no_pca_compute_max_dist_cluster(x_train_data, n_clusters):
    clustering_kmeans = KMeans(n_clusters=n_clusters)
    clustering_kmeans.fit(x_train_data)
    centroids = clustering_kmeans.cluster_centers_
    dist_mat = pd.DataFrame(distance_matrix(x_train_data, centroids))
    max_dist_mat = []
    for i in range(dist_mat.shape[1]):
        max_dist_mat.append(np.max(dist_mat.iloc[:, i]))
    return max_dist_mat, centroids, clustering_kmeans


def compute_max_dist_cluster(model, x_train_data):
    model.fit(x_train_data)
    centroids = model.cluster_centers_
    dist_mat = pd.DataFrame(distance_matrix(x_train_data, centroids))
    max_dist_mat = []
    for i in range(dist_mat.shape[1]):
        max_dist_mat.append(np.max(dist_mat.iloc[:, i]))
    return max_dist_mat, centroids


def detect_normal(x_new_data, centroids, max_dist_mat):
    unknown_index = []
    dist_mat = pd.DataFrame(distance_matrix(x_new_data, centroids))
    for k in range(len(max_dist_mat)):
        for i in range(dist_mat.shape[0]):
            if max_dist_mat[k] < dist_mat.iloc[i, k]:
                unknown_index.append(i)
    return unknown_index


def olindda_normality(x_train_data, n_clusters, x_test_data):
    x_train_data = pd.DataFrame(x_train_data)
    max_dist_mat, centroids, clustering_kmeans = no_pca_compute_max_dist_cluster(x_train_data, n_clusters)
    unknown_index = detect_normal(x_test_data, centroids, max_dist_mat)
    return unknown_index, clustering_kmeans, centroids


def generate_unknown_k_means(x_test_data, unknown_index, n_clusters):
    unknown_index = np.unique(unknown_index)
    unknown_values_x = [x_test_data.loc[i, :] for i in unknown_index]
    unknown_values_x = pd.DataFrame(unknown_values_x)
    model_unknown_kmeans = KMeans(n_clusters=n_clusters)
    max_dist_mat, centroids, clustering_kmeans = no_pca_compute_max_dist_cluster(unknown_values_x, n_clusters)
    return model_unknown_kmeans, max_dist_mat, centroids, unknown_values_x


def compute_overall_mean(model_unknown_kmeans, global_values):
    max_dist_mat_unknown, centroids_unknown = compute_max_dist_cluster(model_unknown_kmeans, global_values)
    dist_mat = pd.DataFrame(distance_matrix(global_values.values, centroids_unknown))
    mean_distances = dist_mat.groupby(dist_mat.idxmin(axis=1)).mean()
    return mean_distances.mean().mean()


def olindda_identification(x_train_data, x_test_data, unknown_index, n_clusters):
    if len(np.unique(unknown_index)) >= n_clusters:
        model_unknown_kmeans, max_dist_mat, centroids, unknown_values_x = \
            generate_unknown_k_means(x_test_data, unknown_index, n_clusters)
        index_list = []
        global_values = pd.concat([x_train_data, x_test_data], axis=0)
        overall_global_mean = compute_overall_mean(model_unknown_kmeans, global_values)
        max_dist_mat_unknown, centroids_unknown = compute_max_dist_cluster(model_unknown_kmeans, unknown_values_x)
        max_dist_mat_unknown = pd.DataFrame(distance_matrix(unknown_values_x.values, centroids_unknown))
        mean_distances_unknown = max_dist_mat_unknown.groupby(max_dist_mat_unknown.idxmin(axis=1)).mean()
        mean_by_cluster_unknown = pd.DataFrame(mean_distances_unknown.mean(axis=1))
        for k in range(mean_by_cluster_unknown.shape[0]):
            if overall_global_mean > np.array(mean_by_cluster_unknown.iloc[k, :]):
                index_list.append(k)
    else:
        overall_global_mean = 0
        mean_by_cluster_unknown = 0
        model_unknown_kmeans = 0
        index_list = [0]
        centroids_unknown = 0
    return overall_global_mean, mean_by_cluster_unknown, model_unknown_kmeans, index_list, centroids_unknown


def compute_global_centroid(old_centroids):
    old_centroids = pd.DataFrame(old_centroids, columns=range(0, old_centroids.shape[1]),
                                 index=range(0, len(old_centroids)))
    centroid_list = old_centroids.T
    global_centroid = pd.DataFrame(np.mean(old_centroids, axis=1)).T
    return centroid_list, global_centroid


def compute_dmax(centroids, global_centroid):
    dist_mat = pd.DataFrame(distance_matrix(centroids, global_centroid))
    d_max = []
    for i in range(dist_mat.shape[1]):
        d_max.append(np.max(dist_mat.iloc[:, i]))
    return d_max


def detect_concept_drift(old_centroids, new_centroids, index_list):
    centroid_list, global_centroid = compute_global_centroid(old_centroids)
    d_max = compute_dmax(centroid_list, global_centroid)
    global_centroid = global_centroid.T
    new_centroids = pd.DataFrame(new_centroids)
    for index in index_list:
        centroids_unknown_temp = new_centroids.iloc[index, :]
        centroids_unknown_temp = pd.DataFrame(centroids_unknown_temp.T)
        d_temp = compute_dmax(centroids_unknown_temp, global_centroid)
        if d_temp <= d_max:
            print("Concept Drift detected for clusters", index)
            has_data_drifted = True
        else:
            print("Novelty identified for clusters", index)
            has_data_drifted = True
        return has_data_drifted


def olindda_detect_drift(x_train_data, x_test_data, n_clusters):
    unknown_index, clustering_kmeans, centroids_known = olindda_normality(x_train_data, n_clusters, x_test_data)
    overall_global_mean, mean_by_cluster_unknown, model_unknown_kmeans, k, centroids_unknown = \
        olindda_identification(x_train_data, x_test_data, unknown_index, n_clusters)
    if len(k) > 1:
        has_data_drifted = detect_concept_drift(centroids_known, centroids_unknown, k)
    else:
        print("No drift or novelty detected in batch")
        has_data_drifted = False
    return has_data_drifted


def drift_detector_generate_nb_sample(data_reference: pd.DataFrame,
                                      data_to_compare: pd.DataFrame,
                                      nb_sample: int = 100):
    """
    Finds the minimum sample size
    """
    assert set(data_reference.columns) == set(data_to_compare.columns)
    nb_sample = np.min([nb_sample, len(data_reference), len(data_to_compare)])
    return nb_sample


def drift_detector_with_multivariate_test(data_reference: pd.DataFrame,
                                          data_to_compare: pd.DataFrame,
                                          n_clusters: int = 3,
                                          nb_days: int = 10,
                                          nb_sample: int = 100) -> list:
    """
    Compare each variable of the two dataframes
    """
    nb_sample = drift_detector_generate_nb_sample(data_reference, data_to_compare, nb_sample)
    list_experiment_results = []
    for _ in range(nb_days):
        data_reference_sample = data_reference.sample(n=nb_sample)
        data_to_compare_sample = data_to_compare.sample(n=nb_sample)
        results = compare_two_dataframes_with_multivariate_test(data1=data_reference_sample,
                                                                data2=data_to_compare_sample,
                                                                n_clusters=n_clusters)
        list_experiment_results.append(results)
    return list_experiment_results


def aggregate_results_of_all_experiments(list_experiment_results: List[Dict]) -> dict:
    column_names = list_experiment_results[0].keys()
    detector_results = {}
    for column_name in column_names:
        experiment_result_for_column_name = [column_results[column_name] for column_results in list_experiment_results]
        is_column_drifted = (np.mean(experiment_result_for_column_name) >= .5)
        detector_results.update({column_name: is_column_drifted})
    return detector_results


def compare_two_dataframes_with_multivariate_test(data1: pd.DataFrame, data2: pd.DataFrame,
                                                  n_clusters: int = 3) -> bool:
    has_data_drifted = olindda_detect_drift(data1, data2, n_clusters)
    return has_data_drifted


def olindda_gradual_drift(data_train: pd.DataFrame,
                          data_to_compare: pd.DataFrame,
                          column_name: str,
                          n_clusters: int,
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
        nb_sample = drift_detector_generate_nb_sample(data_train,
                                                      dataset_corrupted,
                                                      nb_sample=nb_sample)
        data_reference_sample = pd.DataFrame(data_train.sample(n=nb_sample))
        data_reference_sample = data_reference_sample.reset_index(drop=True)
        data_to_compare_sample = pd.DataFrame(dataset_corrupted.sample(n=nb_sample))
        data_to_compare_sample = data_to_compare_sample.reset_index(drop=True)
        olindda_detect_drift(x_train_data=data_reference_sample,
                             x_test_data=data_to_compare_sample,
                             n_clusters=n_clusters)
        days += 1
        value_drift += init_value_drift


def olindda_seasonal_drift(data_train: pd.DataFrame,
                           data_to_compare: pd.DataFrame,
                           column_name: str,
                           n_clusters: int,
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
        nb_sample = drift_detector_generate_nb_sample(data_train,
                                                      dataset_corrupted,
                                                      nb_sample=nb_sample)
        data_reference_sample = data_train.sample(n=nb_sample)
        data_to_compare_sample = dataset_corrupted.sample(n=nb_sample)
        olindda_detect_drift(x_train_data=data_reference_sample,
                             x_test_data=data_to_compare_sample,
                             n_clusters=n_clusters)
        days += 1
