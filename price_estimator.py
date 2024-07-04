import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from loguru import logger as log
import sql_interpreter as si
import tensorflow as tf

# =============================================================================
#                         Basic overview visualisation
# =============================================================================


# Plot the cost of homes
def cost_distribution(price: pd.Series) -> None:
    """ Plot the cost distribution of homes.
    Args:
        price (pd.Series): Series of home prices.
    Returns:
        None """
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


def houses_map(df: pd.DataFrame) -> None:
    """ Create an interactive map showing sold houses with plotly.
    Args:
        df (pd.DataFrame) of house data.
    Returns:
        None """
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


def houses_heatmap(df: pd.DataFrame) -> None:
    """Create an interactive heatmap showing sold houses.
    Args:
        df (pd.DataFrame) of house data.
    Returns:
        None """
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
#                                Regression
# =============================================================================

def regression_model(df: pd.DataFrame) -> None:
    """Build and evaluate a linear regression model.
    Args:
        df (pd.DataFrame) of house data.
    Returns:
        None """
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
    log.info(f'Training data is {train_pct:.3}% of the total data.')
    test_pct = 100*X_test.shape[0]/features.shape[0]
    log.info(f'Test data makes up the remaining {test_pct:0.3}%.')

    # Fit a regression model to beds, baths and parking
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    rsquared = regr.score(X_train, y_train)
    log.info(f'Training data r-squared: {rsquared:.2}')

    # Find regression coefficients, predicted values and residuals
    regr_coef = pd.DataFrame(data=regr.coef_,
                             index=X_train.columns,
                             columns=['Coefficient'])
    log.info(f"Coefficients are: {regr_coef}")
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


def log_regression(df: pd.DataFrame) -> None:
    """Build and evaluate a logarithmic regression model.
    Args:
        df (pd.DataFrame): DataFrame containing house data.
    Returns:
        None """
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

    log.info(f'Training data r-squared: {log_rsquared:.2}')

    df_coef = pd.DataFrame(data=log_regr.coef_,
                           index=X_train.columns,
                           columns=['coef'])
    log.info(f"Dataframe coef: {df_coef}")

    # Distribution of Residuals (log prices) - checking for normality
    log_resid_mean = round(log_residuals.mean(), 2)
    log_resid_skew = round(log_residuals.skew(), 2)

    sns.displot(log_residuals, kde=True, color='navy')
    plt.title(f'Log price model: Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')  # noqa
    plt.show()

    # Make np array of results
    average_vals = np.array(features.mean().values)

    property_stats = pd.DataFrame(
        data=average_vals.reshape(1, len(features.columns)),
        columns=features.columns)

    # Make prediction of an average property
    log_estimate = log_regr.predict(property_stats)[0]
    est = np.e**log_estimate
    dollar_est = "{:.0f}".format(est)
    log.info(f'An average property is estimated to be worth ${dollar_est:.6}')

    # Make an estimate for a specific property
    input_features = {'features.beds': 4,
                      'features.baths': 2,
                      'features.parking': 2,
                      'crime_score': 45,
                      'edu_score': 35}
    pred_df = pd.DataFrame(input_features, index=['index_label'])
    log_pred = log_regr.predict(pred_df)[0]
    dollar_pred = np.e**log_pred
    dollar_pred = "{:.0f}".format(dollar_pred)
    log.info(
        f'The specified property is estimated to be worth ${dollar_pred:.6}')


def train_and_evaluate_gbr(df: pd.DataFrame) -> GradientBoostingRegressor:
    """ Use more powerful Gradient Boosting Regressor model.
    Args:
        df (pd.DataFrame).
    Returns:
        GradientBoostingRegressor: Trained Gradient Boosting Regressor model.
        """
    # Define target variable and features
    print(df.columns)
    target = df['price']
    features = df[['features.beds',
                   'features.baths',
                   'features.parking',
                   'crime_score',
                   'edu_score']]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        random_state=10)

    # Initialize the Gradient Boosting Regressor model
    gbr = GradientBoostingRegressor(n_estimators=100,
                                    learning_rate=0.1,
                                    max_depth=3,
                                    random_state=10)

    # Train the model
    gbr.fit(X_train, y_train)

    # Make predictions on training and testing data
    y_train_pred = gbr.predict(X_train)
    y_test_pred = gbr.predict(X_test)

    # Evaluate model performance
    log.info(f"Training R^2 score: {r2_score(y_train, y_train_pred):.2f}")
    log.info(f"Testing R^2 score: {r2_score(y_test, y_test_pred):.2f}")
    log.info(f"Testing RMSE: {np.sqrt(mean_squared_error(
        y_test, y_test_pred)):.2f}")

    # Feature importance
    feature_importances = pd.Series(gbr.feature_importances_,
                                    index=features.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    log.info(f"Feature importances:\n{feature_importances}")

    # Visualize the results
    visualize_gbr_results(y_train, y_train_pred, y_test, y_test_pred)

    # Example house features for prediction
    house_features = {
        'features.beds': 4,
        'features.baths': 2,
        'features.parking': 2,
        'crime_score': 45,
        'edu_score': 35
    }
    house_features_df = pd.DataFrame(house_features, index=[0])

    # Predict the house price
    predicted_price = gbr.predict(house_features_df)[0]
    log.info(f"The predicted price for the house is: ${predicted_price:.2f}")

    # Calculate residual skew
    residuals = y_test - y_test_pred
    skew = residuals.skew()
    log.info(f"Residual skew is {skew}")

    return gbr


def visualize_gbr_results(y_train, y_train_pred, y_test, y_test_pred):
    """Visualize the results of the model predictions.
    Args:
        y_train (pd.Series): Actual training prices.
        y_train_pred (np.ndarray): Predicted training prices.
        y_test (pd.Series): Actual testing prices.
        y_test_pred (np.ndarray): Predicted testing prices.
    Returns:
        None """
    # Plot Actual vs. Predicted Prices
    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred,
                color='blue',
                alpha=0.6,
                label='Training Data')
    plt.scatter(y_test,
                y_test_pred,
                color='red',
                alpha=0.6,
                label='Testing Data')
    plt.plot([min(y_train.min(), y_test.min()),
              max(y_train.max(), y_test.max())],
             [min(y_train.min(), y_test.min()),
              max(y_train.max(), y_test.max())],
             color='black', linestyle='--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.show()

    # Residual plot for training data
    residuals_train = y_train - y_train_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_train_pred,
                residuals_train,
                color='blue',
                alpha=0.6,
                label='Training Data')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices (Training Data)')
    plt.legend()
    plt.show()

    # Residual plot for testing data
    residuals_test = y_test - y_test_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test_pred,
                residuals_test, color='red',
                alpha=0.6,
                label='Testing Data')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices (Testing Data)')
    plt.legend()
    plt.show()


# =============================================================================
#                         Tensorflow machine learning
# =============================================================================


df = si.sql_query()


def tensorflow_model(df: pd.DataFrame) -> None:
    """Build and evaluate a tensorflow model.
    Args:
        df (pd.DataFrame) of house data.
    Returns:
        None """
    target = df['price']
    features = df.loc[:, ['features.beds',
                          'features.baths',
                          'features.parking',
                          'crime_score',
                          'edu_score']]

    # Ensure the data types are correct
    features = features.astype(float)
    target = target.astype(float)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42
                                                        )

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(features.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot the actual vs predicted prices and save the plot to a file
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.plot([min(y_test), max(y_test)],
             [min(y_test), max(y_test)],
             color='purple')
    plt.savefig("actual_vs_predicted_prices.png")
    print("Plot saved as actual_vs_predicted_prices.png")


tensorflow_model(df)
