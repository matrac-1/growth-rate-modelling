#WIP
#Specific Growth Rate Modelling Using Polynomial and kNN Techniques
The objective of this project was to explore the use of polynomial and k nearest neighbours (kNN) machine learning techniques for predicting bioprocess behavior. Specifically, specific growth rate of a biological process was modelled, based on system parameters. The project uses an implementation (from scikit-learn) of grid search cross-validation to train both kinds of models on the raw data. Polynomials of degree 1 to 8 were studied, as well as k nearest neighbour models with k ranging from 1 to 14 with both uniform and distance-based weights.
##Highlights
•	Applied grid search cross-validation to train polynomial and kNN models.
•	Identified “ideal” / ”optimal” model amongst over 30 considered alternatives .
•	Generated validation curves and RMSE data on each model to illustrate and support “ideal” model conclusion.
•	Analyzed the effect of key process features (temperature, pH, lactate levels, glucose levels) on specific growth rate using the chosen model.
•	Explored the use of scikit-learn libraries (pipelines, GridSearchCV, train_test_split, StandardScaler, PolynomialFeatures, metrics, etc.) for the first time.
##Environment (TBD)
##Code Design
The following section describes the workflow of the Python code in main.py
###Data Loading and Preprocessing (ln 13-28)
Empirical data from dataset_cell_growth_4factor.csv needs to first be accessed and organized before it can be used to train and test the models. The .csv file is first loaded into a pandas data  frame df. The columns of the data frame containing the features and targets are identified (ln 17-18) and then used to create Numpy arrays that act as matrices containing the feature (array x) and target (array y) data. In this case, the first column of the .csv containing the specific growth rate data is passed into the target array, and the remaining columns are passed into the features array.
Next, the data needs to be split into testing and training sets. To do this, train_test_split() from sklearn.model_selection is used. The arrays x and y are fed to train_test_split() and split into 4 new arrays, x_train, x_test, y_train, y_test. An 80/20 train-to-test ratio was used. random_state=0 was also used, to ensure the same splits of data is used across Python installations, making the code give the same outputs no matter where the code is run.
###Polynomial Model Creation and Fitting (ln 31-52)
Once the data is properly treated, the different machine learning models can be implemented. The first of these is the polynomial regression model, polymodel. make_pipline() was used to streamline the model creation workflow, applying 3 different functions in series to the input data. 
First, StandardScaler() is passed into the pipeline, ensuring that the inputted feature data is on a similar, comparable scale, making thee effect of each feature on the specific growth rate much easier to analyse and interpret later. Next, PolynomialFeatures() is used to create additional polynomial features not in the original dataset, such as squared terms (ex. T^2) and interaction terms (ex. T * Glucose). This is what allows the relationship between the features and target to be modelled with polynomial terms while still using a linear regression model. Of note, the polynomial degree is not yet specified at this stage, as it will be varied and “optimized” in the following code. 
Once the model is created, a dictionary, param_grid, is created containing each of the hyperparameters to be tested. In this case, param_grid is passed a range of integers from 1 to 7 inclusively, each acting as a polynomial degree to be tested. The range of degrees/hyperparameters can be modified to the end user’s desire, however metric scores quite poor for the model past degree 7, therefore degrees 8 and beyond are no longer included to reduce the amount of unnecessary data generated.
Finally, GridSearchCV() from the scikit-learn library is used to train the model, using a grid search cross validation algorithm defined by poly_gscv(). Param_grid() is passed to GridSearchCV(). 2 evaluation metrics are calculated and used by GridSearchCV() to judge which polynomial model is most effective, negative RMSE as well as the coefficient of determination. RMSE is used as the main scoring metric, while R2 is only supplemental. 4 folds are specified to be used, as this makes each of the sample splits represent 20% of the original data set, if the initial testing set created with train_test_split() is considered the “5th” fold.
The training data is then fed to poly_gscv() using .fit(x_train, y_train). The results from the model fitting are sent to a pandas data frame using pd.DataFrame(poly_gscv.cv_results_) and further processed into another data frame containing only the relevant RMSE and R2 scoring results necessary for analysing the fitting. This new data frame, summary_poly, is then exported to a .csv file. 
###kNN Model Creation and Fitting (ln 59-83)
The workflow used for generating the kNN models is practically the same as the one described for the polynomial models, with only a few differences that need to be considered.
First, when creating the pipleline, PolynomialFeatures() and LinearRegression() have been obviously swapped KNeighboursRegressor() from scikit-learn. Next, param_grid2 not only ranges the hyperparameters k to be tested from 1 to 14 inclusively, but also now includes a key for the weights (distance versus uniform) to be used. Finally, the coefficient of determination is not used as a scoring metric, since it is not applicable to kNN models. 
###Data Visualisation (ln 88-181)
While long, the remaining parts of the code are straight forward and serve mainly to illustrate the results of the model fitting. 
Lines 88 to 98 print to the console negative RMSE and, for the polynomial model, R2 results for the best fitting polynomial and kNN models using the training data. Lines 101 to 107 do the same, but using the test data set created at the beginning of the program.
Lines 112 to 119 are used to visualise the scaled coefficients and their associated features for the “ideal” polynomial model. This time, the model is first fitted to the entire data set, and its coefficient and feature combinations are extracted as arrays using: 
•	feature_names = poly_gscv.best_estimator_.named_steps["polynomialfeatures"].get_feature_names_out(features),
•	coefs = poly_gscv.best_estimator_.named_steps["linearregression"].coef_ 
•	intercept = poly_gscv.best_estimator_.named_steps["linearregression"].intercept_
A for-loop is then used to iterate over the arrays and print out the coefficients and feature combinations. 
Finally, lines 121 to 181 are used to make the validation plots. For both models, relevant RMSE scores for the training and validation folds were extracted from the results_poly and results_kNN data frames and converted to Numpy arrays to make the information possible to visualize. Matplotlib library functions were used to generate the plots. 
##Analysis 
The following section analyses the results generated by the program using the base data set and code configuration as is presented.
###Grid Search Cross Validation Results for kNN and Polynomial Models
Polynomial and kNN models were employed in the program, as both are relatively simple to understand, interpret, and implement. Before the effectiveness of each model type can be compared to each other however, an “optimized” version of each needs to be found. As such, grid search cross validation was employed to find the optimal hyperparameters for each model, in this case the polynomial degree and k number of neighbours. Negative RMSE was used as the scoring metric. Initially, a very wide range of degrees and k were tested, spanning from 1 to 20 for degrees and 1 to 30 for k, however these ranges were shortened to what you see below since it became quite clear that the RMSE scores became very poor at the higher hyperparameters. The tables below illustrate the results for the grid search CV on both the polynomial and kNN models.
|Poly. degree|RMSE               |RMSE Std. dev.      |R²                 |R² Std. dev.       |
|------------|-------------------|--------------------|-------------------|-------------------|
|1           |0.17387851917345115|0.008007540752697643|0.08407935168477038|0.0708195781256037 |
|2           |0.15422913640597968|0.010392561716728447|0.2829774970723773 |0.01782096021503101|
|3           |0.16452155132164392|0.017973912275773593|0.18714033868560043|0.06898673803033184|
|4           |0.22891394889404865|0.05801221412864946 |-0.6756561676543846|0.8894646898349732 |
|5           |0.6042549009766762 |0.08803074746933981 |-10.244985324642752|3.283714336405476  |
|6           |10.56776236190717  |1.997877333463198   |-3414.5163079367953|1054.9145557478    |
|7           |8.949106742439792  |1.6195680841818214  |-2521.5482490529853|971.9951106808246  |
|k  |Weights            |RMSE                |RMSE Std. dev.     |
|---|-------------------|--------------------|-------------------|
|1  |uniform            |0.18544349399419247 |0.01116346100520698|
|1  |distance           |0.18544349399419247 |0.01116346100520698|
|2  |uniform            |0.1659570637239977  |0.011412842288710827|
|2  |distance           |0.16317158397252973 |0.01187442438524997|
|3  |uniform            |0.16666013862696316 |0.012902828365259555|
|3  |distance           |0.1631769597157273  |0.012379136203243912|
|4  |uniform            |0.16103673635548627 |0.010328323388899663|
|4  |distance           |0.15882613330444667 |0.01084652952123386|
|5  |uniform            |0.1600120422797651  |0.012732338610420935|
|5  |distance           |0.15744033184887357 |0.012600287403544193|
|6  |uniform            |0.1552851198711507  |0.01346739612012451|
|6  |distance           |0.15347662163165898 |0.013116870324186385|
|7  |uniform            |0.15393479916776168 |0.01175614947530805|
|7  |distance           |0.15213189290707774 |0.01164368823950868|
|8  |uniform            |0.15496450071141868 |0.011098415453030693|
|8  |distance           |0.15296584277831374 |0.0111279842022161 |
|9  |uniform            |0.1532708586523278  |0.01130102525546506|
|9  |distance           |0.15154949304542978 |0.011232894347816698|
|10 |uniform            |0.15377451882286136 |0.010873981476379906|
|10 |distance           |0.15186947346546603 |0.011076374829092921|
|11 |uniform            |0.15460230398067776 |0.010104711338038717|
|11 |distance           |0.15245092530927495 |0.01028918962426505|
|12 |uniform            |0.15620726734161336 |0.011429763606428993|
|12 |distance           |0.15350432139390674 |0.011433236766017102|
|13 |uniform            |0.15624737040297343 |0.012986896599049854|
|13 |distance           |0.15348143994813845 |0.01258253314531752|
|14 |uniform            |0.15634077580792138 |0.013387759998916367|
|14 |distance           |0.15352914106891077 |0.01295022125791734|

As can be seen above, the polynomial model possessing the best (lowest) RMSE score has a degree of 2. For kNN, a k-value of 9 provides the best score, and it is also interesting to note that for all values for k that were tested, distance weighting provided a better score than uniform weighting across the board. These conclusions are further supported by the generated validation curves for each of the models.
https://github.com/matrac-1/growth-rate-modelling/blob/main/figures/validation_curve_kNN.png
https://github.com/matrac-1/growth-rate-modelling/blob/main/figures/validation_curve_poly.png
