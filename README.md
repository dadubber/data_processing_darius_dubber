# Dataprocessing movement assessor

This repository contains a functional workflow based on snakemake to train the algorithms random forest, svm, naive 
bayes and K-nearest neighbors on accelerometer data of wearables for the prediction of movements.
This is based on my graduation project, the graduation project can be found at
[link to movement_assessor repository] (https://github.com/dadubber/movement_recognition). The goal of that project was
to create a model that was able to detect basic movements based on the publicly available dataset 
[Heterogeneity activity recognition data set] (https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition).

## Repository structure
```bash
src
├── __init__.py
├── python_scripts
│   ├── create_confusion_matrix.py
│   ├── feature_extraction.py
│   ├── feature_selection_tools.py
│   ├── __init__.py
│   ├── plot_most_important_features.py
│   ├── prepare_data.py
│   ├── preprocess_data.py
│   ├── __pycache__
│   │   └── feature_selection_tools.cpython-36.pyc
│   ├── validate_algorithm.py
│   └── tune_hyper_parameters.py
├── rules
│   ├── common.smk
│   ├── evaluate_algorithm.smk
│   ├── preprocessing.smk
│   ├── tune_hyper_parameters.smk
│   └── validate_algorithm.smk
├── schemas
│   └── config.schema.yaml
└── Snakefile

```

## Usage
The most important folder can be found in the src folder.
This folder contains the created python scripts, the snakemake rules, the snakefile and the config schema.
<br>
To start the workflow it is first necessarily to create a virtual environment and install all the necessary python 
packages that are given in requirements.txt. This can be done with the commands below:<br>
```bash
pip install virtualenv
virtualenv data_processing
source data_processing/venv/bin/activate
pip install -r requirements.txt
```
If all the packages are installed, the workflow can be started with the command below: <br>
```bash
snakemake --snakefile src/Snakefile -c4
```
The argument -c4 can be changed if there are more threads available. This command will start the Snakefile.
A dag file of the workflow is shown below: <br>
![Alt text](dag.png)
In the dag image you can see that first the training data will be downloaded and preprocessed.
During the preprocessing step the data is segmented in segments of a length of 2 seconds.
The only classes taken in account of the heterogeneity dataset are the movements sit, stand, walk, stairs up and stairs
down.

## Reports

## Config

## Results