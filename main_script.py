import domain_scraper_2 as ds2  # noqa
import sql_interpreter as si
import price_estimator as pe  # noqa
import suburbscores as s_score  # noqa
import asyncio  # noqa

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


# ------------------------- Add Suburb Data   ------------------------------- #
df = si.sql_query()  # Get database from SQL server
# df = s_score.run(houses_df=df)  # Add suburb database into existing one


# ------------------------- Analysis   ------------------------------- #
# # General global analysis
# pe.cost_distribution(df.price)  # Frequency distribution of all prices
# pe.houses_map(df)  # Scatter map of ACT with locations of all sold houses
# pe.houses_heatmap(df)  # House map with density plot

# Regression models of price based on beds, baths, and parking
# pe.regression_model(df)
# pe.log_regression(df)


# Most powerful gradient boosting model which extracts important features
pe.train_and_evaluate_gbr(df)
