import json  # type: ignore


def read_suburbs_and_postcodes(file_path):
    suburbs_with_postcodes = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_postcode = None
        for line in lines:
            line = line.strip()
            if line.isdigit():
                current_postcode = line
                suburbs_with_postcodes[current_postcode] = []
            elif line and current_postcode:
                suburbs_with_postcodes[current_postcode].append(line)
    return suburbs_with_postcodes


def save_suburbs_with_postcodes(suburbs_with_postcodes, output_file):
    with open(output_file, 'w') as file:
        json.dump(suburbs_with_postcodes, file, indent=4)


suburbs_with_postcodes = read_suburbs_and_postcodes('suburbs.txt')

save_suburbs_with_postcodes(suburbs_with_postcodes, "suburbs_postcodes.json")
