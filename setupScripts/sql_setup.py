# This file served to create an initial sql database for house price project
# It does not need to be accessed after initialisation

import mysql.connector  # type: ignore
import os  # type: ignore
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()


# Create SQL server database
def create_sql_connection(passwd, host="localhost", user="root"):
    """Takes password, host and username. Default host is localhost,
    default user is root, passwd must be provided."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            passwd=passwd,
            auth_plugin='mysql_native_password',
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection


# Execute sql query
def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


# Connect to an existing SQL server database
def sql_connection(passwd: str, db_name: str,
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


# # Create house price database
create_database_query = "CREATE DATABASE houses"

connection = create_sql_connection(passwd=os.getenv("SQL_PW"))
execute_query(connection=connection, query=create_database_query)
