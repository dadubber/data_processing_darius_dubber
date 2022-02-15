# rule for training the algorithms with the tuned parameters with 10-fold cross validation
rule train_algorithm:
    input:
        'output/labels.csv',
        'output/features.csv',
        '{algorithm}_tuned_parameters.sav'
    output:
        '10_fold_csv_results_{algorithm}.sav',
        'most_important_features_{algorithm}.csv'
    log:
        'logs/train_algorithm_{algorithm}.log'
    script:
        '../python_scripts/train_algorithm.py'