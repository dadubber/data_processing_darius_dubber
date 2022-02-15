# import statements
import glob

import pandas as pd
from snakemake.io import expand
from snakemake.utils import validate

# Validate the config file
# validate(config, schema='../schemas/config.yaml')

# validate the given accelerometer data
# validate(data, schema='../schemas/acc_data.schema.yaml')

def get_final_output():
    """
    Function for creating a list with the final output files
    :return:
    """
    # obtain the output images of the important features for each algorithm
    final_output = expand('output/{algorithm}_important_features.png', algorithm=config["algorithms"])

    # obtain the output images of the confusion matrices and add them to the final output
    final_output + expand('output/{algorithm}_confusion_matrix.png', algorithm=config["algorithms"])

    return final_output
