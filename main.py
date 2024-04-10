import domain_scraper_2  # noqa
import sql_interpreter  # noqa
import price_estimator as pe  # noqa

import asyncio  # noqa

# Set global variables
SQL_DATABASE = 'houses'
SQL_TABLE = 'houses'

# ------------------- Scrape data and write to sql table -------------------- #

# Scrape 1000 entries from domain.com and write to json file
# asyncio.run(domain_scraper_2.run())

# Update the local MySQL server and save data to sql table.
# Note that the mysql server must be created already, and SQL_PW set in .env.
# sql_interpreter.run(table=SQL_TABLE, db=SQL_DATABASE)

# ------------------------- Data Visualisation ------------------------------ #
# df = pe.sql_to_df(table=SQL_TABLE, db_name=SQL_DATABASE)
# df = pe.clean_df(df)


# General global analysis
# pe.cost_distribution(df.price)  # Frequency distribution of all prices
# pe.houses_map(df)  # Scatter map of ACT with locations of all sold houses
# pe.houses_heatmap(df)  # House map with density plot

# # Regression models of price based on beds, baths, and parking
# # pe.regression_model(df)
# pe.log_regression(df)
