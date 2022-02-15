rule create_confusion_matrix:
    input:
        '10_fold_csv_results_{algorithm}.sav'
    output:
        'output/{algorithm}_confusion_matrix.png'
    log:
        'logs/create_confusion_matrix_{algorithm}.log'
    script:
        '../python_scripts/create_confusion_matrix.py'

rule plot_most_important_features:
    input:
        'most_important_features_{algorithm}.csv'
    output:
        'output/{algorithm}_important_features.png'
    log:
        'logs/plot_most_important_features_{algorithm}.log'
    script:
        '../python_scripts/plot_most_important_features.py'