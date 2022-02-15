# Rule to download and unzip the data
rule get_data:
    input:
        config['data_link']
    output:
        config['data_folder']
    log:
        "logs/get_data.log"
    shell:
        'wget {input} -o {output}.zip && unzip -d {output} {output}.zip >{log} 2>&1'

# Rule to prepare the data for preprocessing
rule prepare_data:
    input:
        config['data_folder']
    output:
        'output/data.sav'
    log:
        "logs/prepare_data.log"
    script:
        "../python_scripts/prepare_data.py"

# Rule for extracting the features and labels from the data
rule preprocess_data:
    input:
        'output/data.sav'
    output:
        'output/labels.csv',
        'output/features.csv'
    log:
        "logs/preprocess_data.log"
    script:
        "../python_scripts/preprocess_data.py"

