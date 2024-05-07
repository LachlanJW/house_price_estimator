import pandas as pd


# Create a dataframe of crime by suburb
with open('ReferenceData/suburbcrime.csv', 'r') as file:
    data = file.readlines()
    # Convert the list of strings into a list of tuples
    suburbs_data = [tuple(suburb.strip().replace('(ACT)', '').split(','))[:-1] for suburb in data]

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(suburbs_data, columns=['Suburb', 'Rating', 'Incidents'])

    print(df)