# This script takes data from the domain scraper to create an SQL database
# of house data for a given subrub and number of pages to scrape

import mysql.connector  # type: ignore
import os  # type: ignore
from mysql.connector import Error
from dotenv import load_dotenv
import pandas as pd  # type: ignore
from sqlalchemy import create_engine  # type: ignore

load_dotenv()


# Connect to an existing SQL server database
def create_sql_connection(passwd: str, db_name: str,
                          host: str = "localhost", user: str = "root"):
    """Takes an sql database, password, host and username.
    Default host is localhost, default user is root,
    passwd and database must be provided."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            passwd=passwd,
            auth_plugin='mysql_native_password',
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection


# Execute sql query
def execute_query(connection, query: str):
    """Takes a connection from mysql connector
    and an SQL query to connect to a database"""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


# # Create house data table in houses database
house_table_query = """
CREATE TABLE houses (
    id INT PRIMARY KEY,
    lat FLOAT,
    lon FLOAT,
    date VARCHAR(50),
    time VARCHAR(50),
    severity VARCHAR(50)
    );
"""

pwd = os.getenv("SQL_PW")
connection = create_sql_connection(passwd=pwd, db_name="houses")
execute_query(connection=connection, query=house_table_query)
