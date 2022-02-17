# rule for tuning the hyper parameters of the different algorithms
rule tune_hyper_parameters:
    input:
        features = 'output/Activity_recognition_exp_Watch_accelerometer_features_train.csv',
        labels = 'output/Activity_recognition_exp_Watch_accelerometer_labels_train.csv'
    output:
        'output/Activity_recognition_exp_Watch_accelerometer_tuned_{algorithm}.sav',
        report('output/best_params_{algorithm}.txt', caption='report/results1.rst', category="Step 11",
            subcategory='{algorithm}')
    benchmark:
        'benchmarks/tune_hyper_parameters_Activity_recognition_exp_Watch_accelerometer_{algorithm}.txt'
    threads: 4
    log:
        'logs/tune_hyper_parameters_Activity_recognition_exp_Watch_accelerometer_{algorithm}.log'
    params:
        clf_name = '{algorithm}',
        hyper_parameters = lambda wildcards: config['algorithms']['{}'.format(wildcards.algorithm)]
    script:
        '../python_scripts/tune_hyper_parameters.py'
