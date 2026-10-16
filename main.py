import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
#next part of code pertains to doing grid search cross-validation to find "optimal" polynomial model
#

#data is split into train and test sets. train set will be used to do grid search cross-validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#model is created. polynomial features' degree is undefined to be optimized later
polymodel = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())

#grid of hyperparameters (poly. degrees) to try
param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 8)])

#gscv model is made and fitted. neg. RMSE and r^2 and found and rmse is used for the fitting metric of the model.
#hyperparameter model performances
poly_gscv = GridSearchCV(polymodel, param_grid=param_grid, cv=4, scoring={"rmse": "neg_root_mean_squared_error","r2": "r2"}, refit="rmse",return_train_score=True)
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

#this code saves gridsearch cv scoring results into tables in a.csv format
summary_poly.to_csv(os.path.join("tables", "poly_grid_search_CV_results.csv"), index=False)

#
#next part of code pertains to doing grid search cross-validation to find "optimal" kNN model
#

#model is created.
kNNmodel = make_pipeline(StandardScaler(), KNeighborsRegressor())

#grid of hyperparameters (poly. degrees) to try
param_grid2 = {"kneighborsregressor__n_neighbors": range(1, 15), "kneighborsregressor__weights": ["uniform", "distance"]}

#gscv model is made and fitted. neg. RMSE and r^2 and found and rmse is used for the fitting metric of the model.
#hyperparameter model performances
kNN_gscv = GridSearchCV(kNNmodel, param_grid=param_grid2, cv=4, scoring={"rmse": "neg_root_mean_squared_error"}, refit="rmse", return_train_score=True)
kNN_gscv.fit(x_train, y_train)

#results extraction for analysis of poly. model. results are first "converted" to pandas df.
results_kNN = pd.DataFrame(kNN_gscv.cv_results_)

#useful columns of results are put in its own df
summary_kNN = pd.DataFrame({
    "k": results_kNN["param_kneighborsregressor__n_neighbors"],
    "Weights": results_kNN["param_kneighborsregressor__weights"],
    "RMSE": -results_kNN["mean_test_rmse"],
    "RMSE Std. dev.": results_kNN["std_test_rmse"],
})

#this code saves gridsearch cv scoring results into tables in a.csv format
summary_kNN.to_csv(os.path.join("tables", "knn_grid_search_CV_results.csv"),index=False)

#next set of code shows "best" model params. estimations based on gridsearchCV results.
#RMSE is used to select "best" model for each model type.

print()
print("BEST POLYNOMIAL MODEL")
print(f"Degree      : {poly_gscv.best_params_['polynomialfeatures__degree']}")
print(f"CV RMSE     : {-poly_gscv.best_score_:.3f} +/-{results_poly.loc[poly_gscv.best_index_, 'std_test_rmse']:.3f}")
print(f"CV R²       : {results_poly.loc[poly_gscv.best_index_, 'mean_test_r2']:.3f} +/-{results_poly.loc[poly_gscv.best_index_, 'std_test_r2']:.3f}")
print()
print("BEST kNN MODEL")
print(f"k            : {kNN_gscv.best_params_['kneighborsregressor__n_neighbors']}")
print(f"Weights      : {kNN_gscv.best_params_['kneighborsregressor__weights']}")
print(f"CV RMSE      : {-kNN_gscv.best_score_:.3f} +/-{results_kNN.loc[kNN_gscv.best_index_, 'std_test_rmse']:.3f}")
print()

#next code gets scores for each "best" model on the test fold
print("POLY. SCORE RESULTS ON TEST FOLD")
print(f"Poly. negative RMSE on test data: {poly_gscv.score(x_test, y_test):.3f} [1/d]")
print(f"kNN R²  on test data            : {r2_score(y_test, poly_gscv.predict(x_test)):.3f}")
print()
print("KNN SCORE RESULTS ON TEST FOLD")
print(f"kNN negative RMSE on test data: {kNN_gscv.score(x_test, y_test):.3f} [1/d]")
print()

#code displays coefficients+powers for poly. model fitted on whole data set.
#important to note that these are still scaled, so they cant be "used" directly for prediction
# fit best polynomial model on full dataset
poly_gscv.fit(x, y)
feature_names = poly_gscv.best_estimator_.named_steps["polynomialfeatures"].get_feature_names_out(features)
coefs = poly_gscv.best_estimator_.named_steps["linearregression"].coef_
intercept = poly_gscv.best_estimator_.named_steps["linearregression"].intercept_
print("FINAL POLYNOMIAL MODEL FITTED ON ENTIRE DATA SET")
print(f"{intercept:.6f} (intercept)")
for coef, name in zip(coefs, feature_names):
    print(f"{coef:.6f} * {name}")

#plots validation curves for polynomial model. only up to degree 4 to make it legible
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12
plot_n = 4

degrees = results_poly["param_polynomialfeatures__degree"].to_numpy()
rmse_train = -results_poly["mean_train_rmse"].to_numpy()
rmse_test = -results_poly["mean_test_rmse"].to_numpy()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

ax.plot(degrees[:plot_n], rmse_train[:plot_n], label="Train (Poly)", color=COLORS[0], linestyle="-")
ax.plot(degrees[:plot_n], rmse_test[:plot_n], label="Validation (Poly)", color=COLORS[0], linestyle="--")
ax.set_xlabel("Polynomial Degree", fontsize=fontsize_xy_label)
ax.set_ylabel(" Specific Growth Rate RMSE (1/d)", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(MultipleLocator(1))

fig.savefig(os.path.join("figures", "validation_curve_poly.png"))
plt.close(fig)

#plots validation curves for kNN model
plot_k = 14

results_uniform = results_kNN[results_kNN["param_kneighborsregressor__weights"] == "uniform"]
results_distance = results_kNN[results_kNN["param_kneighborsregressor__weights"] == "distance"]

k_uniform = results_uniform["param_kneighborsregressor__n_neighbors"].to_numpy()
rmse_train_uniform = -results_uniform["mean_train_rmse"].to_numpy()
rmse_valid_uniform = -results_uniform["mean_test_rmse"].to_numpy()

k_distance = results_distance["param_kneighborsregressor__n_neighbors"].to_numpy()
rmse_train_distance = -results_distance["mean_train_rmse"].to_numpy()
rmse_valid_distance = -results_distance["mean_test_rmse"].to_numpy()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

ax.plot(k_uniform[:plot_k], rmse_train_uniform[:plot_k], label="Train (Uniform)", color=COLORS[0], linestyle="-")
ax.plot(k_uniform[:plot_k], rmse_valid_uniform[:plot_k], label="Validation (Uniform)", color=COLORS[0], linestyle="--")

ax.plot(k_distance[:plot_k], rmse_train_distance[:plot_k], label="Train (Distance)", color=COLORS[1], linestyle="-")
ax.plot(k_distance[:plot_k], rmse_valid_distance[:plot_k], label="Validation (Distance)", color=COLORS[1], linestyle="--")

ax.set_xlabel("Number of Neighbors (k)", fontsize=fontsize_xy_label)
ax.set_ylabel("Specific Growth Rate RMSE (1/d)", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=2, fontsize=fontsize_legend)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(MultipleLocator(1))

fig.savefig(os.path.join("figures", "validation_curve_kNN.png"))
plt.close(fig)

