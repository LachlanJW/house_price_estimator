import unittest
import pytest

import json
import os

# =============================================================================
#                              domain_scraper2.py
# =============================================================================
from domain_scraper_2 import pick_and_remove_suburb, parse_search_page


def test_pick_and_remove_suburb():
    # Sample suburb data for testing
    json_file = "suburbs_postcodes_temp.json"
    suburb_data = {
        "2612": ["Turner", "Acton"],
        "2606": ["Lyneham", "O'Connor"]
    }

    # Write the sample suburb data to the temporary JSON file
    with open(json_file, 'w') as file:
        json.dump(suburb_data, file)

    # Call the function under test
    suburb, postcode = pick_and_remove_suburb(json_file)

    # Read the updated data from the JSON file
    with open(json_file, 'r') as file:
        updated_data = json.load(file)

    # Check the selected suburb has been removed from the JSON file
    assert suburb not in updated_data[postcode]

    # Check length of suburb list for the selected postcode is decremented
    assert len(updated_data[postcode]) == 1

    # Clean up: remove the temporary JSON file
    os.remove(json_file)


def test_parse_search_page():
    sample_data = {
        "listingsMap": {
            "1": {"id": 1, "listingModel": {"price": "$555,000", "address": "123 Main St", "features": ["garage"], "tags": {"tagText": "New"}}},  # noqa
            "2": {"id": 2, "listingModel": {"price": "$600,000", "address": "456 Oak St", "features": ["pool"], "tags": {"tagText": "Featured"}}}  # noqa
        }
    }

    # Call the function with the sample data
    parsed_data = parse_search_page(sample_data)

    # Check if the parsed data is correct
    assert len(parsed_data) == 2
    assert parsed_data[0] == {"id": 1, "price": "$555,000", "address": "123 Main St", "features": ["garage"], "date": "New"}  # noqa
    assert parsed_data[1] == {"id": 2, "price": "$600,000", "address": "456 Oak St", "features": ["pool"], "date": "Featured"}  # noqa


if __name__ == "__main__":
    pytest.main([__file__])
