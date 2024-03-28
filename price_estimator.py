import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from setupScripts.sql_interpreter import get_data_from_sql
from typing import Dict, List, Tuple

# Obtain price and room data from SQL and clean up
data = get_data_from_sql(col1="listingModel.price",
                             col2="listingModel.features.beds",
                             table='houses')
price = [float(item[0].replace('$', '').replace(',', '')) for item in data]
rooms = [int(item[1]) for item in data]


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


# Plot number of rooms
def nr_rooms(rooms, price):

    df = pd.DataFrame(zip(rooms, price))

    sns.pairplot(df, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
    plt.show()


# Price vs area
def area_trend():
    data2 = get_data_from_sql(col1="listingModel.price",
                              col2="listingModel.features.landSize",
                              table='houses')
    price = [float(item[0].replace('$', '').replace(',', '')) for item in data2]
    area = [int(item[1]) for item in data2]
    df = pd.DataFrame(zip(area, price))
    filtered_df = df[df[0] != 0]  # exclude apartments with no land
    sns.pairplot(filtered_df, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
    plt.show()


area_trend()
