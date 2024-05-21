import json  # type: ignore


def read_suburbs_and_postcodes(file_path):
    """ Read suburbs and postcodes from a text file
    and store them in a dictionary."""
    suburbs_with_postcodes = {}  # Initialize an empty dictionary

    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_postcode = None

        for line in lines:
            line = line.strip()
            if line.isdigit():  # Check if the line represents a postcode
                current_postcode = line
                suburbs_with_postcodes[current_postcode] = []  # Create list
            # Remove administrative suburbs and append suburb to list
            elif line and current_postcode and "Administrative Postcode Only" not in line:  # noqa
                suburbs_with_postcodes[current_postcode].append(line)
    # Remove postcodes with empty suburb lists
    suburbs_with_postcodes = {postcode: suburbs for postcode, suburbs in suburbs_with_postcodes.items() if suburbs} # noqa
    return suburbs_with_postcodes  # Return the dictionary


def save_suburbs_with_postcodes(suburbs_with_postcodes, output_file):
    with open(output_file, 'w') as file:
        json.dump(suburbs_with_postcodes, file, indent=4)


suburbs_with_postcodes = read_suburbs_and_postcodes('suburbs.txt')

save_suburbs_with_postcodes(suburbs_with_postcodes, "suburbs_postcodes.json")
