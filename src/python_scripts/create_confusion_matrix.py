import pickle
import sys

# logging
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sys.stderr = open(snakemake.log[0], 'w')


def create_plot_confusion_matrix(cnf_matrix, labels, algorithm, data_set, output_file):
    # get labels in alphabetical order
    alphabetical_labels = sorted(set(labels['Movement']))

    # initialize the heatmap
    ax = sns.heatmap(cnf_matrix, annot=True, cmap='Blues')

    # Create the title
    ax.set_title('Confusion matrix of {}'.format(algorithm, data_set))

    # Set the labels for the axis
    ax.set_xlabel('Predicted movements')
    ax.set_ylabel('Actual movements')

    # Set ticket labels
    ax.xaxis.set_ticklabels(alphabetical_labels)
    ax.yaxis.set_ticklabels(alphabetical_labels)

    # Save figure to output file
    plt.savefig(output_file)
    return


def main():
    # obtain the created confusion matrix
    cnf_matrix_file_name = snakemake.input[0]
    with open(cnf_matrix_file_name, 'rb') as input_file:
        cnf_matrix = pickle.load(input_file)
    # obtain the labels
    labels = pd.read_csv(snakemake.input[1])

    # obtain the output file
    output_file = snakemake.output[0]

    # obtain the name of the classifier
    classifier = snakemake.params['algorithm']
    data_set = snakemake.params['data_set']

    # plot the confusion matrix
    create_plot_confusion_matrix(cnf_matrix, labels, classifier, data_set, output_file)


if __name__ == '__main__':
    sys.exit(main())
