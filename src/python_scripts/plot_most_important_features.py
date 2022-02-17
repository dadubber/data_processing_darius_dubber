import sys

# logging
sys.stderr = open(snakemake.log[0], 'w')

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_most_important_features(most_important_features, classifier, data_set, output_file):
    most_important_features_df = pd.DataFrame({'Features': list(most_important_features.loc[0].index),
                                               'Permutation importance': list(most_important_features.loc[0])})
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x='Features', y='Permutation importance', ci=None,
                     data=most_important_features_df.nlargest(n=5, columns=['Permutation importance']))
    ax.set_title('Top 5 important features of {}\n'.format(data_set, classifier))
    ax.set_xlabel('\nFeatures')
    ax.set_ylabel('Permutation importance')
    plt.savefig(output_file)


def main():
    # obtain most important features data frame
    most_important_features_df = pd.read_csv(snakemake.input[0])

    # obtain the output file
    output_file = snakemake.output[0]

    # obtain the name of the classifier
    classifier = snakemake.params['algorithm']
    data_set = snakemake.params['data_set']

    plot_most_important_features(most_important_features_df, classifier, data_set, output_file)


if __name__ == '__main__':
    sys.exit(main())
