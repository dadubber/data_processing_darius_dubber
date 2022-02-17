# rule for training the algorithms with the tuned parameters with 10-fold cross validation
rule validate_algorithm:
    input:
        'output/Activity_recognition_exp_Watch_accelerometer_labels_test.csv',
        'output/Activity_recognition_exp_Watch_accelerometer_features_test.csv',
        'output/Activity_recognition_exp_Watch_accelerometer_tuned_{algorithm}.sav'
    output:
        report('output/{algorithm}_results.txt'),
        '/{algorithm}_confusion_matrix.sav',
        'output/{algorithm}_most_important_features.csv'
    benchmark:
        'benchmarks/train_algorithm_{algorithm}.txt'
    threads: 4
    log:
        'logs/train_algorithm_{algorithm}.log'
    script:
        '../python_scripts/train_algorithm.py'