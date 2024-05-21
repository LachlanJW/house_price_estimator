from unittest.mock import patch

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from house_price_estimator.suburbscores import (clean_education_data,
                                                compile_school_atars,
                                                predict_atar_results)


def convert_df_to_int32(df: pd.DataFrame) -> pd.DataFrame:
    '''Ensure int types in dataframes used are 32 to avoid 32-64 test errors'''
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].astype('int32')
    return df


def test_clean_education_data():
    # Sample raw data
    raw_data = {
        'Column1': ['A', 'B', 'C'],
        'Column2': [1, 2, 3],
        'Median ATAR': ['70', '80', '90'],
        'ATAR >= 65': ['50%', '1200%', '75%']
    }
    df = pd.DataFrame(raw_data)

    # Expected cleaned data
    expected_data = {
        'Median ATAR (2023)': [70, 80, 90],
        'ATAR >= 65 (2023)': [50, 12, 75]
    }
    expected_df = pd.DataFrame(expected_data)

    # Run the function
    result_df = clean_education_data(df, 2023)

    # Ensure data types are ok
    result_df = convert_df_to_int32(result_df)
    expected_df = convert_df_to_int32(expected_df)

    # Compare the result with the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df)


# Sample data for each year
def mock_scrape_table(url):
    year = url.split('=')[-1]
    data = {
        '2008': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['70', '80'], 'ATAR >= 65': ['50%', '60%']}),
        '2009': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['75', '85'], 'ATAR >= 65': ['55%', '65%']}),
        '2010': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['72', '82'], 'ATAR >= 65': ['52%', '62%']}),
        '2011': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['74', '84'], 'ATAR >= 65': ['54%', '64%']}),
        '2012': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['76', '86'], 'ATAR >= 65': ['56%', '66%']}),
        '2013': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['78', '88'], 'ATAR >= 65': ['58%', '68%']}),
        '2014': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['77', '87'], 'ATAR >= 65': ['57%', '67%']}),
        '2015': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['79', '89'], 'ATAR >= 65': ['59%', '69%']}),
        '2016': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['80', '90'], 'ATAR >= 65': ['60%', '70%']}),
        '2017': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['81', '91'], 'ATAR >= 65': ['61%', '71%']}),
        '2018': pd.DataFrame({'College': ['School A', 'School B'], 'Median ATAR': ['82', '92'], 'ATAR >= 65': ['62%', '72%']}),
    }
    return data[year]


def mock_clean_education_data(df, year):
    return df.rename(columns={'Median ATAR': f'Median ATAR ({year})',
                              'ATAR >= 65': f'ATAR >= 65 ({year})'})


# Mocking scrape_table and clean_education_data
@patch('suburbscores.scrape_table', side_effect=mock_scrape_table)
@patch('suburbscores.clean_education_data',
       side_effect=mock_clean_education_data)
def test_compile_school_atars(mock_scrape, mock_clean):
    # Expected merged DataFrame
    expected_data = {
        'College': ['School A', 'School B'],
        'Median ATAR (2008)': [70, 80],
        'ATAR >= 65 (2008)': [50, 60],
        'Median ATAR (2009)': [75, 85],
        'ATAR >= 65 (2009)': [55, 65],
        'Median ATAR (2010)': [72, 82],
        'ATAR >= 65 (2010)': [52, 62],
        'Median ATAR (2011)': [74, 84],
        'ATAR >= 65 (2011)': [54, 64],
        'Median ATAR (2012)': [76, 86],
        'ATAR >= 65 (2012)': [56, 66],
        'Median ATAR (2013)': [78, 88],
        'ATAR >= 65 (2013)': [58, 68],
        'Median ATAR (2014)': [77, 87],
        'ATAR >= 65 (2014)': [57, 67],
        'Median ATAR (2015)': [79, 89],
        'ATAR >= 65 (2015)': [59, 69],
        'Median ATAR (2016)': [80, 90],
        'ATAR >= 65 (2016)': [60, 70],
        'Median ATAR (2017)': [81, 91],
        'ATAR >= 65 (2017)': [61, 71],
        'Median ATAR (2018)': [82, 92],
        'ATAR >= 65 (2018)': [62, 72]
    }
    expected_df = pd.DataFrame(expected_data)

    # Run the function
    result_df = compile_school_atars()

    # Ensure data types are ok
    result_df = convert_df_to_int32(result_df)
    expected_df = convert_df_to_int32(expected_df)

    # Assert that the DataFrame content is the same
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_predict_atar_results():
    # Sample historical data
    data = {
        'College': ['School A', 'School B'],
        'Median ATAR (2008)': [70, 80],
        'Median ATAR (2009)': [71, 81],
        'Median ATAR (2010)': [72, 82],
        'Median ATAR (2011)': [73, 83],
        'Median ATAR (2012)': [74, 84],
        'Median ATAR (2013)': [75, 85],
        'Median ATAR (2014)': [76, 86],
        'Median ATAR (2015)': [77, 87],
        'Median ATAR (2016)': [78, 88],
        'Median ATAR (2017)': [79, 89],
        'Median ATAR (2018)': [80, 90],
    }
    df = pd.DataFrame(data)

    # Run the function
    result_df = predict_atar_results(df.copy())

    # Manually calculate expected predictions for 2024 using linear regression
    expected_predictions = []
    for index, row in df.iterrows():
        atar_values = [row[f'Median ATAR ({yr})'] for yr in range(2008, 2019)]
        years = np.arange(2008, 2019).reshape(-1, 1)
        regr = LinearRegression()
        regr.fit(years, atar_values)
        predicted_ATAR = int(regr.predict([[2024]])[0])
        expected_predictions.append(predicted_ATAR)

    # Create expected DataFrame
    df['2024_ATAR'] = expected_predictions
    expected_df = df

    # Ensure data types are ok
    result_df = convert_df_to_int32(result_df)
    expected_df = convert_df_to_int32(expected_df)

    # Use pandas assert function to compare DataFrames
    pd.testing.assert_frame_equal(result_df, expected_df)
