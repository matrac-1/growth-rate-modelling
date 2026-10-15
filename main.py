# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

#the following code performs initial data processing. data in .csv is put in a pandas dataframe and targets/features
#are extracted and put im numpy arrays to be used by the rest of the program.
df = pd.read_csv(os.path.join('datasets','dataset_cell_growth_4factor.csv'))

features = df.columns[1:]
targets = df.columns[0]

x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

#
#next part of code pertains to doing grid search cross-validation to find optimal polynomial model
#

#data is split into train and test sets. train set will be used to do grid search cross-validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#model is created. polynomial features' degree is undefined to be optimized later
polymodel = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())

#grid of hyperparameters (poly. degrees) to try
param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 8)])

#gscv model is made and fitted. neg. RMSE and r^2 and found and rmse is used for the fitting metric of the model.
#hyperparameter model performances
poly_gscv = GridSearchCV(polymodel, param_grid=param_grid, cv=4, scoring={"rmse": "neg_root_mean_squared_error","r2": "r2"}, refit="rmse", return_train_score=True)
poly_gscv.fit(x_train, y_train)

#results extraction for analysis of poly. model. results are first "converted" to pandas df.
results_poly = pd.DataFrame(poly_gscv.cv_results_)

#useful columns of results are put in its own df
summary_poly = pd.DataFrame({
    "Poly. degree": results_poly["param_polynomialfeatures__degree"],
    "RMSE": -results_poly["mean_test_rmse"],
    "RMSE Std. dev.": results_poly["std_test_rmse"],
    "R²": results_poly["mean_test_r2"],
    "R² Std. dev.": results_poly["std_test_r2"]
})

print(summary_poly.to_string(float_format='{:.3f}'.format))

#
#next part of code fits data to kNN models (uniform and distance)
#

#model is created.
kNNmodel = make_pipeline(StandardScaler(), KNeighborsRegressor())

#grid of hyperparameters (poly. degrees) to try
param_grid2 = {"kneighborsregressor__n_neighbors": range(1, 15), "kneighborsregressor__weights": ["uniform", "distance"]}

#gscv model is made and fitted. neg. RMSE and r^2 and found and rmse is used for the fitting metric of the model.
#hyperparameter model performances
kNN_gscv = GridSearchCV(kNNmodel, param_grid=param_grid2, cv=4, scoring={"rmse": "neg_root_mean_squared_error","r2": "r2"}, refit="rmse", return_train_score=True)
kNN_gscv.fit(x_train, y_train)

#results extraction for analysis of poly. model. results are first "converted" to pandas df.
results_kNN = pd.DataFrame(kNN_gscv.cv_results_)

#useful columns of results are put in its own df
summary_kNN = pd.DataFrame({
    "k": results_kNN["param_kneighborsregressor__n_neighbors"],
    "Weights": results_kNN["param_kneighborsregressor__weights"],
    "RMSE": -results_kNN["mean_test_rmse"],
    "RMSE Std. dev.": results_kNN["std_test_rmse"],
    "R²": results_kNN["mean_test_r2"],
    "R² Std. dev.": results_kNN["std_test_r2"]
})

print(summary_kNN.to_string(float_format='{:.3f}'.format))

#saves results tables to .csv
summary_poly.to_csv(os.path.join("tables", "summary_poly.csv"), index=False)
summary_kNN.to_csv(os.path.join("tables", "summary_kNN.csv"),index=False)