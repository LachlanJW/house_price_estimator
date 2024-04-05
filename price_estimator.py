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
sql_string = f"mysql+mysqlconnector://root:{SQL_PASSWORD}@localhost:3306/{DATABASE_NAME}"  # noqa
engine = create_engine(sql_string)  # Set echo=True to print to console

# Query to select all data from the table
query = "SELECT * FROM houses;"

# Load the data into a DataFrame using Pandas
df = pd.read_sql(query, con=engine)


# Plot the cost of homes
def cost_distribution(price):

    sns.displot(price,
                bins=30,
                kde=True,
                aspect=2,
                color='#2196f3')

    plt.title('Recent house prices in Coombs and Wright ACT')
    plt.xlabel('Price ($Million)')
    plt.ylabel(f'Nr. of Homes, total = {len(price)}')
    plt.xlim(min(price), max(price))  # Adjust x-axis limits
    plt.ylim(0, )  # Ensure y-axis starts from 0
    plt.margins(0.1)
    plt.tight_layout()
    plt.show()


print(df.columns)
# cost_distribution(df.price)

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
