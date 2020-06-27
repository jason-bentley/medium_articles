## Data
The dataset is available here in the repo for ease, however, the source is here: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ where more information about the variables and data collection can be found.

## PyCaret
To learn more about PyCaret please go to: https://pycaret.org/

## Creating the environment from scratch and for use in Jupyter:

```
conda create -n hd_env python=3.7 pandas numpy seaborn matplotlib
conda activate hd_env
pip install pycaret
pip install TableOne
pip install --user ipykernel
python -m ipykernel install --user --name=hd_env
conda env export > heart_disease_venv.yml
conda deactivate
```

## To set-up from yaml file and create a kernel for use in Jupyter:

```
conda env create -f heart_disease_venv.yml
conda activate hd_env
pip install --user ipykernel
python -m ipykernel install --user --name=hd_env
conda deactivate
```
