# Housing-price-predictor

In this project, I train and test supervised learning models to predict housing prices. The model takes in various features such as living area, no. of bedrooms and bathrooms, no. of floors, condition of the house etc. The model preformed with the R squared score of 0.85 and the 95% interval of the Root Mean Squared Error(RMSE) is (114375.43902275835, 163703.87794189138).

Utilised tools: Scikit-learn, Pandas, Matplotlib, Seaborn

## Problem Statement

The real estate market is one of the most competitive and dynamic markets in the world. Accurate house price prediction is beneficial for prospective homeowners, real estate investors, and government policy makers. However, predicting house prices is a complex task due to the multitude of factors that influence the price of a house, such as its location, size, condition, age, proximity to amenities, and the state of the real estate market at the time of sale.

The goal of this project is to develop a machine learning model that can accurately predict the price of a house based on a set of features. It should also be robust to outliers and missing data, which are common in real estate datasets.

The success of the model will be evaluated based on its prediction accuracy on a held-out test set. The model that minimizes the prediction error will be considered the best model. The project will also aim to interpret the model’s predictions to understand which features are most influential in determining house prices.

## Dataset Description

The dataset can be found here: https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india/data

Dataset Overview:

* Title: House Price Dataset of India1
* Size: 23 columns, 14,620 rows2
* Attributes: Includes details such as ID, date, number of bedrooms, bathrooms, living area, lot area, and more.
* Usability: Rated 8.82 for usability with well-documented and clean data.
* License: Specified within the dataset description.
* Accessibility: Available for download on Kaggle with high engagement and numerous views and downloads.

Dataset Significance:

* Purpose: Provides comprehensive data for real estate price analysis in India.
* Applications: Useful for implementing algorithms and gaining insights into the housing market trends.
* Community Impact: Actively used for learning and research, as indicated by the Kaggle community feedback.

## Data Preprocessing

* Converted data field such as `number of bathrooms` and `number of floors` into integer type.
* Removed unnecessary fields such as `id`, `Date`, `Postal code`.
* Removed columns `Area of the house(excluding basement)` and `Area of the basement` to reduce multicolinearity with `living area`.
* Created Train-Test sets using stratefied sampling on `living area` to avoid sampling bias.
* Deployed data processing pipeline that has steps to replace missing data with the median of the field, and standardize the scale of the fields.

## Model Building

Explored models:
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

The training dataset was used to train the above datasets. The models were then evaluated and cross-validated to find the best-performing model. Further, the hyperparameters of the best-performing model were fine-tuned to find the best combination of hyperparameters using a grid search.

The best-found estimator was trained on the full train dataset and evaluated using the test dataset.

## Model Evaluation

For the comparison of different models, the root mean squared errors(RMSE) of the models were used. For cross-validation, RMSE was calculated and compared.

The final model was evaluated by calculating the final RMSE, the 95% interval for the RMSEs and the R-squared score of the estimator.

## Results and Conclusions

The Random forest regressor performed the best out of the three considered models in terms of RMSE. The best estimator hyperparameters found using grid search are {`max_features`: 11, `n_estimators`: 30}. The most important features are `grade of the house`, `living area`, and `latitude` with feature importances of 0.29, 0.25 and 0.14.

The RMSE of the final estimator evaluated on the test dataset is 141210.30, with the 95% confidence interval for the RMSE (114375.43902275835, 163703.87794189138). The R-squared score of the estimator is 0.85.
