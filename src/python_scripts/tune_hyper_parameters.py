import pickle
import sys

from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd

from feature_selection_tools import CorrelatedFeaturesRemover, ScaleDataFrame, \
    FeaturesWithLowVarianceRemover

CLASSIFIERS_DICT = {
    'random_forest': RandomForestClassifier(),
    'naive_bayes': GaussianNB(),
    'knn': KNeighborsClassifier(p=1, weights='distance'),
    'svm': SVC()
}

# logging
sys.stderr = open(snakemake.log[0], 'w')


def create_clf_pipeline(clf_name: str) -> pipeline.Pipeline:
    """
    Creates a sklearn pipeline based on a given classifier
    :param clf_name: The name of the chosen classifier
    :return: A sklearn pipeline
    """
    pipe = pipeline.Pipeline([
        ('remove_l_v_features', FeaturesWithLowVarianceRemover()),
        ('remove_corr_features', CorrelatedFeaturesRemover(0.8)),
        ('Scaler', ScaleDataFrame()),
        ('clf', CLASSIFIERS_DICT[clf_name])
    ])
    return pipe


def tune_hyper_parameters(clf_name: str, hyper_parameters: dict, labels: pd.Series, features: pd.DataFrame):
    # Create pipeline
    pipe = create_clf_pipeline(clf_name)
    # create gridsearch object
    clf = GridSearchCV(pipe, hyper_parameters, cv=5, n_jobs=1, scoring='f1_weighted')
    # find best params
    clf.fit(features, labels['Movement'])
    return clf


def main():
    # obtain the name of the classifier
    clf_name = snakemake.params['clf_name']
    # obtain the hyper parameters
    hyper_parameters = snakemake.params['hyper_parameters']
    # obtain the labels
    labels = pd.read_csv(snakemake.input['labels'])
    # obtain the features
    features = pd.read_csv(snakemake.input['features'])
    # tune hyper parameters
    clf = tune_hyper_parameters(clf_name, hyper_parameters, labels, features)

    # write best found parameters to output file
    with open(snakemake.output[0], 'wb') as output_file:
        pickle.dump(clf, output_file)
    return


if __name__ == '__main__':
    sys.exit(main())
