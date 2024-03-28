# This script takes data from the domain scraper to create an SQL database

import os  # type: ignore
import pandas as pd  # type: ignore
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text  # type: ignore
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List

load_dotenv()

# Create an sql engine
sql_string = f"mysql+mysqlconnector://root:{os.getenv("SQL_PW")}@localhost:3306/houses"
engine = create_engine(sql_string, echo=True)  # Echo prints sql to console


def json_to_sql_table(data: List[Dict], engine):
    """Take a json object and an sql engine and normalise, create a pandas
    dataframe, then insert as an sql table. Note this is highly inefficient
    and created purely for curiosity"""
    normalise = pd.json_normalize(data)
    dataframe = pd.DataFrame(normalise)

    dataframe.to_sql(name='houses_test', con=engine,
                     if_exists='replace', index=False)

    return


def run_table_maker(engine):
    # Open test data file
    with open("data.json", 'r') as file:
        test_data = json.load(file)

    # Add the data as a table to SQL server, overwrites previous
    json_to_sql_table(data=test_data, engine=engine)

    return


# Get data from sql table
def sql_query(query: str):
    try:
        with engine.connect() as conn:
            sql = text('SELECT * FROM houses_test')
            result = conn.execute(sql)
            rows = result.fetchall()
            for row in rows:
                print(row)
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")


query = 'SELECT * FROM houses_test'
sql_query(query)
