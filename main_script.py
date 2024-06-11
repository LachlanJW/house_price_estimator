import domain_scraper_2 as ds2
import sql_interpreter as si
import price_estimator as pe
import suburbscores as s_score
import asyncio
from loguru import logger as log
import json

# Set global variables
SQL_DATABASE = 'houses'
SQL_TABLE = 'houses'


# ----------------------------- Scraping ------------------------------------ #
# Run this line to add another suburb to the data.json file
# by scraping domain.com. Note: the sql server will not automatically update.
# asyncio.run(ds2.run())


# ---------------------------- SQL Writing ---------------------------------- #
# Run this line to update the local MySQL server and save data to sql table.
# Note that the mysql server must be created already, and SQL_PW set in .env.
# si.run(table=SQL_TABLE, db=SQL_DATABASE)


# ------------------------- Add Suburb Data --------------------------------- #
# df = si.sql_query()  # Get database from SQL server
# df = s_score.run(houses_df=df)  # Add suburb database into existing one


# ------------------------- Analysis ---------------------------------------- #
# # General global analysis
# pe.cost_distribution(df.price)  # Frequency distribution of all prices
# pe.houses_map(df)  # Scatter map of ACT with locations of all sold houses
# pe.houses_heatmap(df)  # House map with density plot

# Regression models of price based on beds, baths, and parking
# pe.regression_model(df)
# pe.log_regression(df)

# Most powerful gradient boosting model which extracts important features
# pe.train_and_evaluate_gbr(df)


def prompt_user():
    """Prompts the user for input and returns their choice."""
    return input('What would you like to do? Enter one of the following:\n'
                 '1 for a frequency distribution of house prices\n'
                 '2 for a heatmap of houses in the database by location\n'
                 '3 to run a regression model and predict house prices\n'
                 'E to exit\n')


def run():
    df = si.sql_query()  # Get database from SQL server

    user_choice = prompt_user()

    # Check if input is valid
    while user_choice not in ['1', '2', '3', 'E']:
        log.error('Invalid user input')
        user_choice = prompt_user()

    # If the user does not exit, run desired functions
    while user_choice != 'E':
        if user_choice == '1':
            pe.cost_distribution(df.price)  # Frequency distribution
        elif user_choice == '2':
            pe.houses_heatmap(df)  # House map with density plot
        elif user_choice == '3':
            pe.train_and_evaluate_gbr(df)  # Regression model
        else:
            log.error('Invalid user input slipped through elif statements')

        user_choice = prompt_user()


if __name__ == "__main__":
    run()
