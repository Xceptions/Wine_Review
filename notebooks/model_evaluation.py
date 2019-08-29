#' % MACHINE LEARNING MODEL EVALUATION
#' % Kenechukwu Agbo
#' % 28th Aug, 2019

#' # Introduction

#' The model was built using boosting as a method and <b>xgboost</b> as a tool
#' Engineered features were saved into a pickle file and used for the feature
#' engineering of the input data. This was done to ensure model persistence.
#' However, this introduced a bit of data leakage into our data, which means
#' that this model will have to be retrained at set intervals with the addition
#' of new set of data. This is because the generated features such as:
#' mean and median will change thereby leading to a change in other generated
#' features.

#' Scaling the data was not done because it will lead to too much of data
#' leakage in our case

#' # LET US START WITH THE MODEL
#' After performing some Exploratory Data Analysis on the dataset,
#' the insight I was able to come up with was that a person's review
#' of wine is based on Price, Culture and Individual differences.

#' Features were then selected based on columns that could reflect these
#' characteristics namely: Price, Country, Province and the target feature
#' "points". New set of data were then generated and used to train the model.
#' Let us see how they reflect on the model

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import warnings
from sklearn.metrics import mean_squared_error as mse

warnings.filterwarnings("ignore")

with open('../data_root/Train/Model.pkl', 'rb') as pkl_model:
    xgb_model = pickle.load(pkl_model)

print(xgb_model.feature_importances_)
train_features = [
    'price',
    'price_per_country_mean',
    'price_per_country_mean_diff',
    'price_per_country_median',
    'price_per_country_median_diff',
    'price_per_province_mean',
    'price_per_province_mean_diff',
    'price_per_province_median',
    'price_per_province_median_diff',
    'points_per_country_mean',
    'points_per_country_median',
    'points_per_province_mean',
    'points_per_province_median'
]
feature_importance = xgb_model.feature_importances_
plt.figure(figsize=(20, 25))
sns.set(font_scale=2)
sns.barplot(x=feature_importance, y=train_features)
plt.show()

#' # PREDICTING THE TEST DATASET
#' we have to take into consideration that some countries appeared
#' only once and if they fell into our test set while spliting, we would
#' not be able to predict their outputs, hence skip them.

predictions = pd.read_csv('../data_root/Evaluations/predictions.csv')
predictions = predictions.iloc[:, 0].values
data = pd.read_csv('../data_root/Evaluations/test_data.csv')
data = data['points'].values

#' # For the test data
plt.title("Distribution of Test Data")
sns.distplot(data, hist=False, rug=True)
plt.show()

#' # For the predictions data
plt.title("Distribution of Predictions")
sns.distplot(predictions, hist=False, rug=True)
plt.show()

#' The mean squared error is
print(mse(data, predictions))

#' A mean squared error of 6.35381 seems very good for our model

#' # Summary
#' Seeing that we could not totally block out data leakage from our data,
#' i.e. we cannot really pinpoint what the true mean will be, however, we
#' could use it but not the standard deviation because that is more
#' sensitive.
#' we would need to keep retraining the model at set intervals
#' using new data that we get.

#' This would ensure a more accurate model


