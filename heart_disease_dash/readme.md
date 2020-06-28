## Data
The dataset is available here in the repo for ease, however, the source is here: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ where more information about the variables and data collection can be found.

## PyCaret
To learn more about PyCaret please go to: https://pycaret.org/

## Creating the environment from scratch and for use with PyCharm:
Note here deliberately doing pycaret and then shap as separate so I can access latest version of shap which is technically not compatible with pycaret

```
conda create -n dash_env python=3.7 pandas numpy seaborn matplotlib
conda activate dash_env
pip install pycaret==1.0.0
pip install shap==0.35.0
pip install dash dash-renderer dash-html-components dash-core-components dash-bootstrap-components dash-daq
conda env export > heart_disease_medium_dash_venv.yml
conda deactivate

```