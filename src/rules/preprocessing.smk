# Rule to download the data
rule get_data:
    output:
        temp("tmp/data/{}.zip".format(config['data_folder']))
    log:
        "logs/get_data_Activity_recognition_exp.log"
    shell:
        "wget {0} -o {1} >{2} 2>&1".format(config['data_link'], '{output}', '{log}')

# Rule to unzip the data
rule unzip_data:
    input:
        "tmp/data/{}.zip".format(config['data_folder'])
    output:
        temp("tmp/data/{}".format(config['data_folder']))
    log:
        "logs/unzip_data_{}.log".format(config['data_folder'])
    shell:
        'unzip -d tmp/data/Activity_recognition_exp tmp/data/Activity_recognition_exp.zip >{log} 2>&1'

# Rule to prepare the data for preprocessing
rule prepare_data:
    input:
        "tmp/data/{}".format(config['data_folder'])
    output:
        'output/Activity_recognition_exp_Watch_accelerometer.sav'
    params:
        acc_filename = config['acc_filename']
    log:
        "logs/prepare_data_Activity_recognition_exp_Watch_accelerometer.log"
    script:
        "../python_scripts/prepare_data.py"

# Rule for preprocessing the data
rule preprocess_data:
    input:
        'output/Activity_recognition_exp_Watch_accelerometer.sav'
    output:
        'output/Activity_recognition_exp_Watch_accelerometer_segments.sav'
    params:
        resampled_frequency = config['resampled_frequency'],
        sampling_frequencies = config['sampling_frequencies']
    log:
        "logs/preprocess_data_Activity_recognition_exp_Watch_accelerometer.log"
    script:
        "../python_scripts/preprocess_data.py"

# Rule for extracting the features
rule extract_features:
    input:
        'output/Activity_recognition_exp_Watch_accelerometer_segments.sav'
    output:
        features_train = 'output/Activity_recognition_exp_Watch_accelerometer_features_train.csv',
        features_test = 'output/Activity_recognition_exp_Watch_accelerometer_features_test.csv',
        labels_train = 'output/Activity_recognition_exp_Watch_accelerometer_labels_train.csv',
        labels_test = 'output/Activity_recognition_exp_Watch_accelerometer_labels_test.csv'
    params:
        resampled_frequency = config['resampled_frequency']
    log:
        "logs/extract_features_Activity_recognition_exp_Watch_accelerometer.log"
    script:
        "../python_scripts/feature_extraction.py"
