## Data
The dataset is available here in the repo for ease, however, the source is here: https://archive.ics.uci.edu/ml/datasets/heart+Disease where more information about the variables and data collection can be found.
This project also follows on from a previous one using Pycaret to develop a predictive model, see
1. [Medium article:](https://towardsdatascience.com/developing-a-model-for-heart-disease-prediction-using-pycaret-9cdf03a66f42) Developing a model for heart disease prediction using PyCaret
2. [Github project:](https://github.com/jason-bentley/medium_articles/tree/master/heart_disease_pycaret) medium_articles/heart_disease_pycaret/

## Set-up conda environment and run dashboard:
To create the conda environment used for the work in this repo and starting up the dashboard, you can do the following:

```
conda env create -f hd_dash_venv.yml
conda activate dash_env
python dashboard_model_example.py
```