# rule for tuning the hyperparameters of the different algorithms
rule tune_hyper_parameters:
    input:
        'output/labels.csv',
        'output/features.csv',
        config['algorithms']
    output:
        '{algorithm}_tuned_parameters.sav', algorithm = config['algorithms']
    log:
        'logs/tune_hyper_parameters_{algorithm}'
    script:
        '../python_scripts/tune_hyper_parameters.py'
