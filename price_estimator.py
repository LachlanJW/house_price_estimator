import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# Load environment variables
load_dotenv()
SQL_PASSWORD = os.getenv("SQL_PW")
DATABASE_NAME = "houses"

# Create SQL engine
sql_string = f"mysql+mysqlconnector://root:{SQL_PASSWORD}@localhost:3306/{DATABASE_NAME}"
engine = create_engine(sql_string)  # Set echo=True to print to console

def get_data_from_sql(table):
    """Retrieve data from an SQL table"""
    try:
        with engine.connect() as conn:
            sql_query = text(f"SELECT price, `features.beds` FROM {table};")
            result = conn.execute(sql_query)
            return result.fetchall()
    except SQLAlchemyError as e:
        print(f"An error occurred while fetching data: {e}")
        return []


# Obtain price and room data from SQL
data = get_data_from_sql('houses')
price = [row[0] for row in data]

# Plot the cost of homes
def cost_distribution(price):

    sns.displot(price,
                bins=30,
                kde=True,
                aspect=2,
                color='#2196f3')

    plt.title(f'Recent house prices in Coombs, ACT')
    plt.xlabel('Price ($Million)')
    plt.ylabel(f'Nr. of Homes, total = {len(price)}')
    plt.xlim(min(price), max(price))  # Adjust x-axis limits based on data range
    plt.ylim(0, )  # Ensure y-axis starts from 0
    plt.margins(0.1)
    plt.tight_layout()
    plt.show()


cost_distribution(price)


# # Plot number of rooms
# def nr_rooms(rooms, price):

#     df = pd.DataFrame(zip(rooms, price))

#     sns.pairplot(df, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
#     plt.show()


# # Price vs area
# def area_trend():
#     data2 = get_data_from_sql(col1="listingModel.price",
#                               col2="listingModel.features.landSize",
#                               table='houses')
#     price = [float(item[0].replace('$', '').replace(',', '')) for item in data2]
#     area = [int(item[1]) for item in data2]
#     df = pd.DataFrame(zip(area, price))
#     filtered_df = df[df[0] != 0]  # exclude apartments with no land
#     sns.pairplot(filtered_df, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
#     plt.show()


