from house_price_estimator.sql_interpreter import clean_df
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    data = {
        'id': [1, 2, None],
        'address.postcode': [12345, None, 67890],
        'features.beds': [3, 2, None],
        'features.baths': [2, None, 1],
        'features.parking': [1, 2, None],
        'features.landSize': [500, None, 750],
        'address.lat': [-33.865143, -33.865143, 40.712776],
        'address.lng': [151.209900, -33.865143, -74.005974],
        'price': ['$1,000,000', '$850,000', '$600,000'],
        'features.propertyType': ['House', 'ApartmentUnitFlat', 'House'],
        'features.propertyTypeFormatted': ['House', 'Flat', 'House'],
        'features.landUnit': ['sqm', 'sqm', 'sqm'],
        'features.isRural': [False, False, False],
        'features.isRetirement': [False, False, False]
    }
    return pd.DataFrame(data)


def test_clean_df(sample_df):
    cleaned_df = clean_df(sample_df)

    expected_data = {
        'id': [1, 2, 0],
        'address.postcode': [12345, 0, 67890],
        'features.beds': [3, 2, 0],
        'features.baths': [2, 0, 1],
        'features.parking': [1, 2, 0],
        'features.landSize': [500, 0, 750],
        'address.lat': [-33.865143, -33.865143, 40.712776],
        'address.lng': [151.209900, -33.865143, -74.005974],
        'price': [1000000, 850000, 600000],
        'features.propertyType': ['House', 'Apartment', 'House']
    }
    expected_df = pd.DataFrame(expected_data)

    # Ensure the expected data types match the cleaned DataFrame
    expected_df = expected_df.astype({
        'id': 'int32',
        'address.postcode': 'int32',
        'features.beds': 'int32',
        'features.baths': 'int32',
        'features.parking': 'int32',
        'features.landSize': 'int32',
        'address.lat': 'float64',
        'address.lng': 'float64',
        'price': 'int32',
        'features.propertyType': 'object'
    })

    pd.testing.assert_frame_equal(cleaned_df, expected_df, check_dtype=True)
