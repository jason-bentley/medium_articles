#!/usr/bin/env bash

# create conda environment from yml file
conda env create -f environment.yml

# activate, add name to venv kernel, and start jupyter
conda activate optuna_env
python -m ipykernel install --user --name optuna_env --display-name "optuna_env"
jupyter notebook