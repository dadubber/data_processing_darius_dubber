import pickle
import sys

import pandas as pd

# logging
sys.stderr = open(snakemake.log[0], 'w')


def read_accelerometer_data(acc_data: str) -> pd.DataFrame:
    """
    Reads in the accelerometer data of the waches
    :param acc_data: path to the accelerometer data
    :return:
    """
    return pd.read_csv(acc_data)


def main():
    # Read in the dataframe
    acc_df = read_accelerometer_data(snakemake.input[0])
    # Save the dataframe in a pickled file
    with open(snakemake.output[0], 'wb') as output_file:
        pickle.dump(acc_df, output_file)


if __name__ == '__main__':
    sys.exit(main())
