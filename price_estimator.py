import os

import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import plotly.express as px  # type: ignore
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sqlalchemy import create_engine

from sql_interpreter import sql_query


# Plot the cost of homes
def cost_distribution(price):

    sns.displot(price,
                bins=30,
                kde=True,
                aspect=2,
                color='#2196f3')

    plt.title('Recent house prices in ACT')
    plt.xlabel('Price ($Million)')
    plt.ylabel(f'Nr. of Homes, total = {len(price)}')
    plt.xlim(min(price), max(price))  # Adjust x-axis limits
    plt.ylim(0, )  # Ensure y-axis starts from 0
    plt.margins(0.1)
    plt.tight_layout()
    plt.show()


# Use plotly to create an interactive map and sold houses on top
def houses_map(df):
    fig = px.scatter_mapbox(df,
                            lat="address.lat",
                            lon="address.lng",
                            hover_name="price",
                            hover_data=["price"],
                            zoom=8,
                            height=800,
                            width=800,
                            mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


# Use plotly to create an interactive heatmap and sold houses on top
def houses_heatmap(df):
    fig = px.density_mapbox(df,
                            lat="address.lat",
                            lon="address.lng",
                            radius=2,
                            hover_name="price",
                            hover_data=["price"],
                            zoom=8,
                            height=800,
                            width=800,
                            mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


# =============================================================================
#                                  Regression
# =============================================================================

def regression_model(df):
    target = df['price']
    features = df.loc[:, ['features.beds',
                          'features.baths',
                          'features.parking',
                          'crime_score',
                          'edu_score']]

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=10
                                                        )

    # % of test training and data set
    train_pct = 100*len(X_train)/len(features)
    print(f'Training data is {train_pct:.3}% of the total data.')
    test_pct = 100*X_test.shape[0]/features.shape[0]
    print(f'Test data makes up the remaining {test_pct:0.3}%.')

    # Fit a regression model to beds, baths and parking
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    rsquared = regr.score(X_train, y_train)
    print(f'Training data r-squared: {rsquared:.2}')

    # Find regression coefficients, predicted values and residuals
    regr_coef = pd.DataFrame(data=regr.coef_,
                             index=X_train.columns,
                             columns=['Coefficient'])
    predicted_vals = regr.predict(X_train)
    residuals = (y_train - predicted_vals)

    # Plot Actual vs. Predicted Prices
    plt.figure(dpi=100)
    plt.scatter(x=y_train, y=predicted_vals, c='indigo', alpha=0.6)
    plt.plot(y_train, y_train, color='cyan')
    plt.title('Actual vs Predicted Prices', fontsize=17)
    plt.xlabel('Actual prices', fontsize=14)
    plt.ylabel('Prediced Prices', fontsize=14)
    plt.show()

    # Residuals vs Predicted values (check if randomly distributed)
    plt.figure(dpi=100)
    plt.scatter(x=predicted_vals, y=residuals, c='indigo', alpha=0.6)
    plt.title('Residuals vs Predicted Prices', fontsize=17)
    plt.xlabel('Predicted Prices', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.show()

    # Residual Distribution Chart
    resid_mean = round(residuals.mean(), 2)
    resid_skew = round(residuals.skew(), 2)

    sns.displot(residuals, kde=True, color='indigo')
    plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
    plt.show()


def log_regression(df):
    new_target = np.log(df['price'])  # Use log prices
    features = df.loc[:, ['features.beds',
                          'features.baths',
                          'features.parking',
                          'crime_score',
                          'edu_score']]

    X_train, X_test, log_y_train, log_y_test = train_test_split(features,
                                                                new_target,
                                                                test_size=0.2,
                                                                random_state=10
                                                                )

    log_regr = LinearRegression()
    log_regr.fit(X_train, log_y_train)
    log_rsquared = log_regr.score(X_train, log_y_train)

    log_predictions = log_regr.predict(X_train)
    log_residuals = (log_y_train - log_predictions)

    print(f'Training data r-squared: {log_rsquared:.2}')

    df_coef = pd.DataFrame(data=log_regr.coef_,
                           index=X_train.columns,
                           columns=['coef'])
    print(df_coef)

    # Distribution of Residuals (log prices) - checking for normality
    log_resid_mean = round(log_residuals.mean(), 2)
    log_resid_skew = round(log_residuals.skew(), 2)

    sns.displot(log_residuals, kde=True, color='navy')
    plt.title(f'Log price model: Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')  # noqa
    plt.show()

    # Make dataframe of results
    average_vals = features.mean().values
    property_stats = pd.DataFrame(data=average_vals.reshape(1, len(features.columns)),  # noqa
                                  columns=features.columns)

    # Make prediction of an average property
    log_estimate = log_regr.predict(property_stats)[0]
    est = np.e**log_estimate
    dollar_est = "{:.0f}".format(est)
    print(f'An average property is estimated to be worth ${dollar_est:.6}')

    # Make an estimate for a specific property
    input_features = {'features.beds': 4,
                      'features.baths': 2,
                      'features.parking': 2,
                      'crime_score': 45,
                      'edu_score':35}
    pred_df = pd.DataFrame(input_features, index=['index_label'])
    log_pred = log_regr.predict(pred_df)[0]
    dollar_pred = np.e**log_pred
    dollar_pred = "{:.0f}".format(dollar_pred)
    print(f'The specified property is estimated to be worth ${dollar_pred:.6}')
