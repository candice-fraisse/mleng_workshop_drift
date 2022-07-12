import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier

def dataset_concat(old_dataset: pd.DataFrame, new_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_dataset = pd.concat([old_dataset, new_dataset], axis=0)
    X_test_corrupted = new_dataset.loc[:, new_dataset.columns != "quality"]
    y_test_corrupted = new_dataset["quality"]
    return X_test_corrupted, y_test_corrupted


def weighted_dataset(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict) -> tuple[any, any]:
    updated_X_train = X_train.copy()
    len_index = updated_X_train.shape[0]+1
    weighted_list = np.array(sorted(range(1, len_index)))
    weight = weighted_list / len_index
    params["weight_column"] =  np.array(weight)
    weighted_lgbm_model = LGBMClassifier(**params)
    weighted_lgbm_model.fit(updated_X_train, y_train)
    return updated_X_train, weighted_lgbm_model


def stacking_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    knn = KNeighborsClassifier(n_neighbors=1)
    rf = RandomForestClassifier(random_state=1)
    gnb = GaussianNB()
    lr = LogisticRegression()
    lgbm = LGBMClassifier(**params)
    estimators = [knn,gnb,rf,lgbm, lr]
    stack = StackingCVClassifier(classifiers = estimators,
                            shuffle = False,
                             use_probas = True,
                             cv = 5,
                             meta_classifier = LogisticRegression())
    stack.fit(X_train, y_train)
    return stack
