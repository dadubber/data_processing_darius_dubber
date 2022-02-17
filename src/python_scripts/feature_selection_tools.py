from collections import defaultdict

from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class MergeDict(dict):
    """
    Class created for merging dicts without overwriting the keys
    """

    def __add__(self, other):
        own = self.copy()
        try:
            for key, val in other.items():
                if key in own:
                    try:
                        own[key] += val
                    except TypeError:
                        # In case of a dict inside dict
                        own[key] = MergeDict(own[key]) + val
                else:
                    own[key] = val
        except AttributeError:
            return NotImplemented
        return MergeDict(own)


class ScaleDataFrame(StandardScaler):
    """

    """
    def transform(self, x, copy=None):
        columns = x.columns
        scaled_data = super().transform(x, copy)
        return pd.DataFrame(scaled_data, columns=columns)

    def fit_transform(self, x, y=None, **fit_params):
        columns = x.columns
        scaled_data = super().fit_transform(x, y=y)
        return pd.DataFrame(scaled_data, columns=columns)


class FeaturesWithLowVarianceRemover(BaseEstimator, TransformerMixin):
    """
    Remove features based on VarianceThreshold from sklearn.
    Instead of VarianceThreshold returns a dataframe
    """

    def __init__(self, threshold=0):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selected_features_names = None
        return

    def fit(self, x, y=None):
        self.selector.fit(x)
        return self

    def transform(self, x, y=None):
        new_x = x.iloc[:, self.selector.get_support()]
        self.selected_features_names = new_x.columns
        return new_x


class CorrelatedFeaturesRemover(BaseEstimator, TransformerMixin):
    """
    A transformer that can be used to remove highly correlated features at a given threshold (standard 0.9)
    """

    def __init__(self, threshold=0.9):
        self.col_corr = []
        self.threshold = threshold
        self.selected_features_names = None
        return

    def fit(self, x, y=None):
        corr_matrix = x.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= self.threshold) and (corr_matrix.columns[j] not in self.col_corr):
                    colname = corr_matrix.columns[i]
                    self.col_corr.append(colname)
        return self

    def transform(self, x, y=None):
        bool_list = [bool(column not in self.col_corr) for column in x.columns.tolist()]
        df = x.iloc[:, bool_list]
        self.selected_features_names = df.columns
        return df


def cal_mean_decreasing_accuracy(x, y, given_classifier, features):
    """
    Calculates the mean decreasing accuracy for each feature with a given classifier
    :param x:
    :param y:
    :param given_classifier:
    :param features:
    :return:
    """
    # Take remaining features from x
    x_used = x[features]
    scores = defaultdict(list)
    # Take ten percentage as test set
    x_train, x_test, y_train, y_test = train_test_split(x_used, y, test_size=0.2, stratify=y)
    # clone classifier
    classifier = clone(given_classifier)
    # train classifier
    classifier.fit(x_train, y_train)
    # predict
    pred_labels_baseline = classifier.predict(x_test)
    baseline_accuracy = accuracy_score(y_test, pred_labels_baseline)
    for i in range(len(features)):
        x_test_cols = x_test.columns
        x_test_arr = x_test.to_numpy()
        save = x_test_arr[:, i].copy()
        x_test_arr[:, i] = np.random.permutation(x_test_arr[:, i])
        per_acc = accuracy_score(y_test, classifier.predict(x_test))
        x_test_arr[:, i] = save
        x_test = pd.DataFrame(x_test_arr, columns=x_test_cols)
        scores[features[i]].append((baseline_accuracy - per_acc) / baseline_accuracy)
    importances = {feature: np.mean(score) for feature, score in scores.items()}
    return importances
