import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression  # type: ignore
from price_estimator import sql_query
from functools import reduce
from geopy.geocoders import Nominatim  # type: ignore
from geopy.distance import geodesic  # type: ignore
from loguru import logger as log


# =============================================================================
#                                Crime Stats
# =============================================================================


def crime_db() -> pd.DataFrame:
    # Create a dataframe of crime by suburb
    with open('ReferenceData/suburbcrime.csv', 'r') as file:
        data = file.readlines()
        # Convert the list of strings into a list of tuples
        suburbs_data = [tuple(suburb.strip().replace('(ACT)', '').split(','))[:-1] for suburb in data]  # noqa

        # Create a DataFrame from the list of tuples
        df = pd.DataFrame(suburbs_data, columns=['Suburb',
                                                 'Rating',
                                                 'Incidents'])
        print(df)
        return df


# =============================================================================
#                              Education Stats
# =============================================================================


def clean_df(df: pd.DataFrame, yr: int) -> pd.DataFrame:
    '''For tables on the bettereducation website, drop irrelevant columns,
    convert numbers to numeric types, and deal with incorrect values.
    Args: pd. Dataframe, yr: int of the year. Returns: pd.Dataframe'''
    # Drop first two columns
    df = df.drop(df.columns[[0, 1]], axis=1)
    # Rename columns
    df = df.rename(columns={'Median ATAR': f'Median ATAR ({yr})',
                            'ATAR >= 65': f'ATAR >= 65 ({yr})'})

    # Clean some of the data in the final column which varies by year
    # Convert the column to numeric
    col = f'ATAR >= 65 ({yr})'
    df[col] = pd.to_numeric(df[col].str.rstrip('%').str.replace(',', ''))
    # If the column is larger than 100, divide by 100
    df.loc[df[col] > 100, col] = (df.loc[df[col] > 100, col] / 100)
    # Convert all values to int
    df[col] = df[col].astype(int)

    # Convert the median ATAR column to float
    col1 = f'Median ATAR ({yr})'
    df[col1] = df[col1].astype(int)

    return df


def scrape_table(url: str) -> pd.DataFrame:
    '''Obtains the first table in a given url and returns a dataframe'''
    # Read HTML tables from the webpage
    tables = pd.read_html(url)

    # Assuming the first table contains the desired data
    return tables[0]


def school_atars() -> pd.DataFrame:
    '''Return a single dataframe of each college in ACT and its ATAR results
    from 2008 to 2018'''
    # Move through the available years of data and create a list of dataframes
    dataframes = []
    for yr in range(2008, 2019):
        # URL of the website to scrape
        url = f"https://bettereducation.com.au/results/ACT.aspx?yr={yr}"

        df = scrape_table(url)
        df = clean_df(df, yr)
        dataframes.append(df)
    # Merge the dataframes based on the school column
    merged_df = reduce(lambda left, right: pd.merge(left,
                                                    right,
                                                    on='College'), dataframes)
    return merged_df


def predict_results(df: pd.DataFrame) -> pd.DataFrame:
    '''Performs a linear regression model on school ATAR history
    to predict scores for 2024'''
    # Add new df column to receive 2024 ATAR prediction
    df['2024_ATAR'] = None

    # Create regression object
    regr = LinearRegression()

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Get ATAR results for the current school
        atar_values = [row[f'Median ATAR ({yr})'] for yr in range(2008, 2019)]
        years = np.arange(2008, 2019).reshape(-1, 1)

        # Fit a model to the ATAR data
        regr.fit(years, atar_values)

        # Predict the 2024 ATAR and add to the DataFrame
        predicted_atar_2024 = int(regr.predict([[2024]])[0])
        df.at[index, '2024_ATAR'] = predicted_atar_2024

    return df


def school_address(df: pd.DataFrame) -> pd.DataFrame:
    '''Uses google geopy to search school names for a latitude and longitude,
    and adds these as columns to the dataframe'''
    # Create new columns to receive lat and lon data
    df['Lat'] = None
    df['Lon'] = None

    # Create a google geolocator
    geolocator = Nominatim(user_agent="school_locator")

    for school in df['College']:
        # Get location for the school
        location = geolocator.geocode(school)

        # Update 'Lat' and 'Long' columns with the returned values
        df.loc[df['College'] == school, 'Lat'] = location.latitude
        df.loc[df['College'] == school, 'Lon'] = location.longitude

    return df


def find_closest_school(house_lat: float, house_lon: float,
                        schools_df: pd.DataFrame) -> int:
    """Find the closest school to a given house based on its
    latitude and longitude. Return its 2024 predicted ATAR."""
    closest_distance = float('inf')
    closest_school_atar = None

    for _, school in schools_df.iterrows():
        school_lat = school['Lat']
        school_lon = school['Lon']
        distance = geodesic((house_lat, house_lon),
                            (school_lat, school_lon)).kilometers
        if distance < closest_distance:
            closest_distance = distance
            closest_school_atar = school['2024_ATAR']

    return closest_school_atar


def update_dataframe(houses_df, schools_df, crime_df):
    """Add the closest school's ATAR, and crime score
    to each house in the houses DataFrame."""
    # Create new columns in houses_df to store values
    houses_df['edu_score'] = None
    houses_df['crime_score'] = None

    # Iterate over each house in houses_df
    for index, house in houses_df.iterrows():
        # Add score for closest school
        house_lat = house["address.lat"]
        house_lon = house["address.lng"]
        edu_score = find_closest_school(house_lat, house_lon, schools_df)
        houses_df.at[index, 'edu_score'] = edu_score
        # Add score for suburb crime
        suburb = house["address.suburb"]
        try:
            rating = crime_df.loc[crime_df['Suburb'].str.lower() == suburb.lower(), 'Rating'].values[0]
        # If suburb does not have crime stats
        except IndexError:
            log.error(f"issue with {suburb}")
        houses_df.at[index, 'crime_score'] = rating
    log.success("Updated dataframe")
    return houses_df


def run(houses_df):
    """Take the existing houses database and includes
    education and crime scores"""
    # Make education score database
    school_df = predict_results(school_atars())
    school_df = school_address(school_df)  # Add the latitude and longitude

    # Make crime database
    crime_df = crime_db()

    # Update values in the hosues df
    houses_df = update_dataframe(houses_df=houses_df,
                                 schools_df=school_df,
                                 crime_df=crime_df)

    return houses_df


if __name__ == "__main__":
    run(houses_df=sql_query())
