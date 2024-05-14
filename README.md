# House Price Estimator
This is a project designed for practice in scraping, data handling and SQL. There are many inefficiencies (conversion of a dataframe to SQL and back) simply conducted for practise.
Scraping from the domain.com website should be conducted at slow rates, of publicly available data, in small quantities.

The **main_script.py** may be used to call the scripts for each part, or they may be run individually from their separate files:

## Part 1: Obtaining house price data
The first portion of this project deals with obtaining house price data from domain.com.au using the scrapfly API. The script, **domain_scraper2.py** searches through result pages in a given postcode for the listing data and creates a **data.json** file of the output. Each time the script is run, output is appended to this file. The **suburbs_postcodes.json** keeps track of which suburbs have already been added to avoid double-ups.


## Part 2: Creating an SQL database of the scraped data
The database setup is performed by **sql_interpreter** which takes the json output of **domain_scraper2.py** and connects to the MySQL database to create (overwrite) the existing json file. Some data cleaning and type conversions are conducted. This needs to be run each time the scraper is run and json updated. This script also contains the functions to retrive data from the SQL table, or overwrite it when more data is needed.

## Part 3: Adding suburb data
Due to poor intial correlation between house features and price, more data was required. Both crime and school data are added by **suburbscores.py**. Crime data is from the AFP website (/ReferenceData/suburbcrime.csv). School data is retrived from the better education website and for the available years 2008 to 2018. The model predicts 2024 school results by linear regression, finds the closest school to each house using geopy (Nominatim for school address lookup, geodesic for distance comparisons), and adds the score to the dataframe. This overwrites the SQL table with the new data. 


## Part 4: Price modelling
The **price_estimator** is the main data analysis script.

Basic data visualisation with histograms and maps may be conducted to see the overall data structure and missing suburbs, or price distributions.

Three models are available for predictions: linear, log, and gradient boosting regression models. These fit house prices against bedrooms, bathrooms, parking, suburb crime and suburb school quality.


## General
**tests.py** describe some basic tests for key functions within the script and will be updated further soon. 
See the **evaluation.md** for a brief discussion of the results of the price modelling and importance of each function.


## Old Scripts
**domain_scraper.py** is an early version of the later script, only looking at a single page of results from the website.
**sql_setup** simply initialised the local sql database and table which is interacted with in part 2.
**suburb_postcode_generator** took suburbs and postcodes copied from the internet into **suburbs.txt** and created the more readable json format used in Part 1.