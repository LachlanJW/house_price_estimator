# This script takes data from the domain scraper to create an SQL database
# Running the script will open data.json and overwrite the sql table houses
# The get_data_from_sql function is called by the price estimator script

import os
import json
from dotenv import load_dotenv
import pandas as pd  # type: ignore
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Sequence

# Load environment variables
load_dotenv()
SQL_PASSWORD = os.getenv("SQL_PW")
DATABASE_NAME = "houses"

# Create SQL engine
sql_string = f"mysql+mysqlconnector://root:{SQL_PASSWORD}@localhost:3306/{DATABASE_NAME}"  # noqa
engine = create_engine(sql_string)  # Set echo=True to print to console


def create_table_from_json(data: List[Dict], table: str = 'houses') -> None:
    """Create an SQL table from JSON data.
    Args: List[Dict] given from json.load.
    Returns: None. The data is written to a local sql table"""
    try:
        # Normalize JSON data and create a DataFrame
        normalized_data = pd.json_normalize(data)
        df = pd.DataFrame(normalized_data)

        df = clean_df(df)  # Perform some data cleaning, type conversion etc

        # Write DataFrame to SQL table
        df.to_sql(name=table, con=engine,
                  if_exists='replace', index=False)
        print("Successfully written to sql table")
    except SQLAlchemyError as e:
        print(f"An error occurred while creating the table: {e}")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Perform some data cleaning specific to this dataframe.
    Args and Returns: pd.Dataframe."""
    # First, convert some columns to integer, some to float
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


def get_data_from_sql(table: str) -> Sequence:
    """Retrieve data from an SQL table"""
    try:
        with engine.connect() as conn:
            sql_query = text(f"SELECT * FROM {table};")
            result = conn.execute(sql_query)
            return result.fetchall()
    except SQLAlchemyError as e:
        print(f"An error occurred while fetching data: {e}")
        return []


def main():
    # Read test data from JSON file
    try:
        with open("data.json", 'r') as file:
            test_data = json.load(file)
    except FileNotFoundError:
        print("Error: data.json file not found.")
        return

    # Create SQL table from JSON data
    create_table_from_json(test_data)


if __name__ == "__main__":
    main()
