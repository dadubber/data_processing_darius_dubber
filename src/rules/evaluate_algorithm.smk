rule plot_confusion_matrix:
    input:
        'output/{algorithm}_confusion_matrix.sav',
        'output/Activity_recognition_exp_Watch_accelerometer_labels_test.csv'
    output:
        report('output/{algorithm}_confusion_matrix.png', caption="report/fig2.rst", category='Step 7',
            subcategory='{algorithm}')
    log:
        'logs/create_confusion_matrix_Activity_recognition_exp_Watch_accelerometer_{algorithm}.log'
    params:
        algorithm = '{algorithm}',
        data_set = 'Heterogeneity Activity Recognition Data Set'
    script:
        '../python_scripts/create_confusion_matrix.py'

rule plot_most_important_features:
    input:
        'output/{algorithm}_most_important_features.csv'
    output:
        report('output/{algorithm}_most_important_features.png', caption="report/fig3.rst", category='Step 8',
            subcategory='{algorithm}')
    log:
        'logs/plot_most_important_features_{algorithm}.log'
    params:
        algorithm='{algorithm}',
        data_set='Heterogeneity Activity Recognition Data Set'
    script:
        '../python_scripts/plot_most_important_features.py'