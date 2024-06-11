# This script takes data from the domain scraper to create an SQL database
# Running the script will open data.json and overwrite the sql table houses
# The get_data_from_sql function is called by the price estimator script

import os
import json
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List
from loguru import logger as log


# Obtain pandas dataframe from SQL server
def sql_query(db_name: str = 'houses',
              query: str = 'SELECT * FROM houses') -> pd.DataFrame:
    """ From a local mysql server take a full table of data and convert
    to a pandas dataframe.
    Args:
        table, db_name.
    Returns:
        pandas dataframe """
    # Get password from local .env file
    load_dotenv()
    SQL_PASSWORD = os.getenv("SQL_PW")

    sql_string = f"mysql+mysqlconnector://root:{SQL_PASSWORD}@localhost:3306/{db_name}"  # noqa
    engine = create_engine(sql_string)  # Set echo=True to print to console

    df = pd.read_sql(query, con=engine)

    return df


def create_table_from_json(engine, data: List[Dict], table: str) -> None:
    """ Create an SQL table from JSON data.
    Args:
        List[Dict] given from json.load.
    Returns:
        None. The data is written to a local sql table """
    try:
        df = clean_df(data)  # Perform some data cleaning, type conversion etc

        # Write DataFrame to SQL table
        df.to_sql(name=table, con=engine,
                  if_exists='replace', index=False)
        log.success("Successfully written to sql table")
    except SQLAlchemyError as e:
        log.error(f"An error occurred while creating the table: {e}")


def clean_df(data: List[Dict]) -> pd.DataFrame:
    """ Perform some data cleaning specific to this dataframe.
    Args:
        pd.Dataframe.
    Returns:
        pd.Dataframe. """
    # Normalize JSON data and create a DataFrame
    normalized_data = pd.json_normalize(data)
    df = pd.DataFrame(normalized_data)

    # Convert some columns to integer, some to float
    int_list = ["id",
                "address.postcode",
                "features.beds",
                "features.baths",
                "features.parking",
                "features.landSize"
                ]
    for column in int_list:  # Fill NA values as 0
        df[column] = df[column].fillna(0).astype(int)

    df["address.lat"] = df["address.lat"].fillna(0).astype(float)
    df["address.lng"] = df["address.lng"].fillna(0).astype(float)

    # Then some string parsing
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
    df['price'] = df['price'].astype(int)
    df.loc[df['features.propertyType'] == 'ApartmentUnitFlat',
           'features.propertyType'] = 'Apartment'

    # Finally, remove some irrelevant columns
    df = df.drop(columns=[
        'features.propertyTypeFormatted',
        'features.landUnit',
        'features.isRural',
        'features.isRetirement'
        ])

    return df


def write_to_sql(df: pd.DataFrame, db_name: str = 'houses'):
    """ Write a pandas DataFrame to a SQL server.
    Args:
        df: pd.DatFrame to be written,
        db_name: str name of the database.
    Returns:
        None """
    # Get password from local .env file
    load_dotenv()
    SQL_PASSWORD = os.getenv("SQL_PW")

    sql_string = f"mysql+mysqlconnector://root:{SQL_PASSWORD}@localhost:3306/{db_name}"  # noqa
    engine = create_engine(sql_string)

    # Write DataFrame to SQL server, overwriting the existing table
    df.to_sql('houses', con=engine, if_exists='replace', index=False)


def run(db: str = 'houses', table: str = 'houses'):
    # Load environment variables
    load_dotenv()
    SQL_PASSWORD = os.getenv("SQL_PW")
    DATABASE_NAME = db

    # Create SQL engine
    sql_string = f"mysql+mysqlconnector://root:{SQL_PASSWORD}@localhost:3306/{DATABASE_NAME}"  # noqa
    engine = create_engine(sql_string)  # Set echo=True to print to console
    # Read test data from JSON file
    try:
        with open("data.json", 'r') as file:
            test_data = json.load(file)
    except FileNotFoundError:
        log.error("Error: data.json file not found.")
        return

    # Create SQL table from JSON data
    create_table_from_json(engine=engine, data=test_data, table=table)


if __name__ == "__main__":
    run()
