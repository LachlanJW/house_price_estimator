# This script takes data from the domain scraper to create an SQL database
# Running the script will open data.json and overwrite the sql table houses_test
# The get_data_from_sql function is called by the price estimator script

import os
import json
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()
SQL_PASSWORD = os.getenv("SQL_PW")
DATABASE_NAME = "houses"

# Create SQL engine
sql_string = f"mysql+mysqlconnector://root:{SQL_PASSWORD}@localhost:3306/{DATABASE_NAME}"
engine = create_engine(sql_string)  # Set echo=True to print to console


def create_table_from_json(data: List[Dict]) -> None:
    """Create an SQL table from JSON data"""
    try:
        # Normalize JSON data and create a DataFrame
        normalized_data = pd.json_normalize(data)
        dataframe = pd.DataFrame(normalized_data)

        # Write DataFrame to SQL table
        dataframe.to_sql(name='houses_test', con=engine,
                         if_exists='replace', index=False)
    except SQLAlchemyError as e:
        print(f"An error occurred while creating the table: {e}")


def get_data_from_sql(col1: str, col2: str, table: str) -> List[Tuple]:
    """Retrieve data from an SQL table"""
    try:
        with engine.connect() as conn:
            sql_query = text(f"SELECT `{col1}`, `{col2}` FROM {table};")
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
