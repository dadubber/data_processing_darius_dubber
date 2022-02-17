import pickle
import sys

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix
import pandas as pd

# logging
from snakemake.utils import report

sys.stderr = open(snakemake.log[0], 'w')


def validate_algorithm(tuned_classifier, test_features, test_labels):
    # predict test set
    pred_labels = tuned_classifier.predict(test_features)
    # Calculate accuracy score
    accuracy = accuracy_score(test_labels, pred_labels)
    # Calculate f1-score
    f1 = f1_score(test_labels, pred_labels, average='weighted')
    # Jaccard score
    jaccard = jaccard_score(test_labels, pred_labels, average='weighted')
    # create confusion matrix
    cnf_matrix = confusion_matrix(test_labels, pred_labels)

    # Obtain feature importances
    perm_importance = permutation_importance(tuned_classifier, test_features, test_labels)
    perm_importance_dict = {k: [perm_importance.importances_mean[i]] for i, k in enumerate(test_features.columns)}
    return accuracy, f1, jaccard, cnf_matrix, perm_importance_dict


def main():
    # obtain test labels
    test_labels = pd.read_csv(snakemake.input[0])

    # obtain test features
    test_features = pd.read_csv(snakemake.input[1])

    # obtain tuned classifier
    tuned_clf_file_name = snakemake.input[2]
    with open(tuned_clf_file_name, 'rb') as input_file:
        tuned_clf = pickle.load(input_file)

    # validate the classifier
    accuracy, f1, jaccard, cnf_matrix, perm_importance = validate_algorithm(tuned_clf, test_features, test_labels)

    # Write obtained scores to given output file
    scores = pd.DataFrame({'scores': ['accuracy', 'f1-score weighted', 'jaccard_score weighted'],
                           'results': [accuracy, f1, jaccard]})
    scores.to_csv(snakemake[0])

    # Write obtained confusion matrix and feature importance to output files
    with open(snakemake.output[1], 'wb') as output_file:
        pickle.dump(cnf_matrix, output_file)

    most_important_features_df = pd.DataFrame(perm_importance)
    most_important_features_df.to_csv(snakemake.output[2])


if __name__ == '__main__':
    sys.exit(main())

