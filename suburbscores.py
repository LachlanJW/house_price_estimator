import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression  # type: ignore
from sql_interpreter import write_to_sql, sql_query
from functools import reduce
from geopy.geocoders import Nominatim  # type: ignore
from geopy.distance import geodesic  # type: ignore
from loguru import logger as log


# =============================================================================
#                                Crime Stats
# =============================================================================


def load_crime_data() -> pd.DataFrame:
    """ Load crime data from a CSV file and return a DataFrame.
    Args:
        None.
    Returns:
        pd.DataFrame: Crime data. """
    # Create a dataframe of crime by suburb
    with open('ReferenceData/suburbcrime.csv', 'r') as file:
        data = file.readlines()
        # Convert the list of strings into a list of tuples
        suburbs_data = [
            tuple(suburb.strip().replace('(ACT)', '').split(','))[:-1]
            for suburb in data
            ]

        # Create a DataFrame from the list of tuples
        df = pd.DataFrame(suburbs_data, columns=['Suburb',
                                                 'Rating',
                                                 'Incidents'])

        return df


# =============================================================================
#                              Education Stats
# =============================================================================


def clean_education_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """ Clean and format education data.
    Args:
        df (pd.DataFrame): Raw education data.
        year (int): Year to iterate through columns.
    Returns:
        pd.DataFrame. """
    # Take and rename relevant columns
    df = df.drop(df.columns[[0, 1]], axis=1)
    df = df.rename(columns={'Median ATAR': f'Median ATAR ({year})',
                            'ATAR >= 65': f'ATAR >= 65 ({year})'})

    # Final column has varying entries which require cleaning and conversion
    col = f'ATAR >= 65 ({year})'
    # Convert string to numeric
    df[col] = pd.to_numeric(df[col].str.rstrip('%').str.replace(',', ''),
                            errors='coerce')
    # If the column is over 100% divide by 100 to handle wierd entries
    df[col] = np.where(df[col] > 100, df[col] / 100, df[col]).astype(int)

    # Convert median ATAR column to int
    col1 = f'Median ATAR ({year})'
    df[col1] = pd.to_numeric(df[col1], errors='coerce').fillna(0).astype(int)

    return df


def scrape_table(url: str) -> pd.DataFrame:
    return pd.read_html(url)[0]


def compile_school_atars() -> pd.DataFrame:
    """ Compile ATAR data for all schools in the ACT from 2008 to 2018.
    Returns:
        pd.DataFrame: DataFrame containing compiled ATAR data. """
    # Make a list of dataframes from the webpage of each years results
    base_url = "https://bettereducation.com.au/results/ACT.aspx?yr="
    dataframes = [
        clean_education_data(scrape_table(f"{base_url}{yr}"), yr)
        for yr in range(2008, 2019)
    ]
    # Merge into a single dataframe
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='College'),
                       dataframes)
    return merged_df


def predict_atar_results(df: pd.DataFrame) -> pd.DataFrame:
    """ Predict 2024 ATAR scores using linear regression with historical data.
    Args:
        df (pd.DataFrame): Historical ATAR data.
    Returns:
        pd.DataFrame: With added 2024 ATAR predictions. """
    df['2024_ATAR'] = None
    regr = LinearRegression()

    for index, row in df.iterrows():
        # Make a list of atars for the available year range (2008 to 2018)
        atar_values = [row[f'Median ATAR ({yr})'] for yr in range(2008, 2019)]
        # Make a list of the years for regression model
        years = np.arange(2008, 2019).reshape(-1, 1)
        # Fit model
        regr.fit(years, atar_values)
        # Predict atar and add to the 2024_ATAR column of the dataframe
        predicted_ATAR = int(regr.predict([[2024]])[0])
        df.at[index, '2024_ATAR'] = predicted_ATAR

    return df


def add_school_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """ Add latitude and longitude coordinates to schools using geopy.
    Args:
        df (pd.DataFrame): School data.
    Returns:
        pd.DataFrame: With added latitude and longitude. """
    # Initiate a Nominatim searcher and add empty rows to dataframe
    geolocator = Nominatim(user_agent="school_locator")
    df['Lat'] = None
    df['Lon'] = None

    # Find school locations and add lat and lon to dataframe
    for index, row in df.iterrows():
        try:
            location = geolocator.geocode(row['College'])
            if location:
                df.at[index, 'Lat'] = location.latitude
                df.at[index, 'Lon'] = location.longitude
        except Exception as e:
            log.error(f"Failed to find {row['College']}: {e}")

    return df


def find_closest_school(house_lat: float, house_lon: float,
                        schools_df: pd.DataFrame) -> int:
    """ Find the closest school to a given house based on latitude
    and longitude.
    Args:
        house_lat (float).
        house_lon (float).
        schools_df (pd.DataFrame): School data.
    Returns:
        int: Predicted 2024 ATAR score of the closest school. """
    # Calculate the distance from the house to each school with apply
    distances = schools_df.apply(
        lambda row: geodesic(
            (house_lat, house_lon), (row['Lat'], row['Lon'])
        ).kilometers, axis=1
    )
    # Find index of lowest score with idxmin()
    closest_index = distances.idxmin()
    # Retrieve the 2024 ATAR score using the index
    closest_school_atar = schools_df.at[closest_index, '2024_ATAR']

    return int(closest_school_atar)


def update_houses_with_scores(houses_df: pd.DataFrame,
                              schools_df: pd.DataFrame,
                              crime_df: pd.DataFrame) -> pd.DataFrame:
    """ Add education and crime scores to each house in the DataFrame.
    Args:
        houses_df (pd.DataFrame).
        schools_df (pd.DataFrame).
        crime_df (pd.DataFrame).
    Returns:
        pd.DataFrame: Updated DataFrame with added scores. """
    # Update the education scores of each house
    houses_df['edu_score'] = houses_df.apply(
        lambda row: find_closest_school(row["address.lat"],
                                        row["address.lng"],
                                        schools_df), axis=1)

    # Update crime scores using apply row-wise.
    # 1: Find where the suburb from crime db matches house address
    # 2: Extract the crime rating value
    # 3. Check the crime db is not empty. If not, return 0
    houses_df['crime_score'] = houses_df.apply(
        lambda row: crime_df[
            crime_df['Suburb'].str.lower() == row["address.suburb"].lower()
        ]['Rating'].values[0]
        if not crime_df[
            crime_df['Suburb'].str.lower() == row["address.suburb"].lower()
        ].empty
        else 0,
        axis=1
    )

    log.success("Updated house dataframe with education and crime scores")
    return houses_df


def clean_house_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean house data by removing outliers and duplicates.
    Args:
        df (pd.DataFrame).
    Returns:
        pd.DataFrame: Cleaned. """
    # Find outliers (z-score > 3)
    z_scores = (df['price'] - df['price'].mean()) / df['price'].std()
    # Remove outliers and drop duplicates
    cleaned_df = df[abs(z_scores) <= 3].drop_duplicates()
    log.info(f"Removed {len(df) - len(cleaned_df)} outliers and duplicates")
    return cleaned_df


def run(houses_df: pd.DataFrame) -> pd.DataFrame:
    """ Main function to update house data with education and crime scores.
    Writes the data back to the sql server for use in main script.
    Args:
        houses_df (pd.DataFrame): DataFrame containing house data.
    Returns:
        pd.DataFrame: Updated house DataFrame. """
    log.success("Starting the update process")

    # Grab school results
    school_df = add_school_coordinates(
        predict_atar_results(compile_school_atars())
        )

    # Grab crime data
    crime_df = load_crime_data()

    # Update and clean dataframe
    updated_houses_df = update_houses_with_scores(houses_df,
                                                  school_df,
                                                  crime_df)
    cleaned_houses_df = clean_house_data(updated_houses_df)

    # Write to sql server
    write_to_sql(cleaned_houses_df)
    log.success("Update process completed")

    return cleaned_houses_df


if __name__ == "__main__":
    houses_df = sql_query()
    run(houses_df)
