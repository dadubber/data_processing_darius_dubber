rule plot_confusion_matrix:
    input:
        'output/Activity_recognition_exp_Watch_accelerometer_{algorithm}_algorithm_results.sav'
    output:
        'output/{algorithm}_confusion_matrix.png'
    log:
        'logs/create_confusion_matrix_Activity_recognition_exp_Watch_accelerometer_{algorithm}.log'
    script:
        '../python_scripts/create_confusion_matrix.py'

rule plot_most_important_features:
    input:
        'most_important_features_{algorithm}.csv'
    output:
        'output/{algorithm}_most_important_features.csv'
    log:
        'logs/plot_most_important_features_{algorithm}.log'
    script:
        '../python_scripts/plot_most_important_features.py'