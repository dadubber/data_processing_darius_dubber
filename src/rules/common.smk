# import statements
from snakemake.io import expand
from snakemake.utils import validate

# Validate the config file
validate(config, schema='../schemas/config.schema.yaml')


def get_final_output():
    """
    Function for creating a list with the final output files
    :return:
    """
    # obtain the output images of the important features for each algorithm
    final_output = expand('output/{algorithm}_confusion_matrix.sav',
        algorithm=config['algorithms'])

    # obtain the output images of the confusion matrices and add them to the final output
    final_output + expand('output/{algorithm}_most_important_features.csv',
        algorithm=config['algorithms'])

    return final_output
