# House Price Estimator
This is a project designed for practice in scraping, data handling and SQL. There are many inefficiencies (conversion of a dataframe to SQL and back) simply conducted for practise.
Scraping from the domain.com website should be conducted at slow rates, of publicly available data, in small quantities.

The **main.py** script may be used to call the scripts for each part, or they may be run individually from their separate files:

## Part 1: Obtaining house price data
The first portion of this project deals with obtaining house price data from domain.com.au using the scrapfly API. The central script, **domain_scraper2.py** searches through result pages in a given postcode for the listing data and creates a **data.json** file of the output. Each time the script is run, output is appended to this file. The **suburbs_postcodes.json** keeps track of which suburbs have been added to avoid double-ups.


## Part 2: Creating an SQL database of the scraped data
The database setup is performed by **sql_interpreter** which takes the json output of **domain_scraper2.py** and connects to the MySQL database to create (overwrite) the existing json file. Some data cleaning and type conversions are conducted. This needs to be run each time the scraper is run and json updated.


## Part 3: Accessing the SQL database
The **price_estimator** is the main data analysis script. It accesses the SQL database to obtain the data and uses seaborn for analysis. To view specific column names, i.e. for a SELECT request, the sql values have been saved as **houses_houses.sql**.

Basic data visualisation with histograms and maps is conducted, then linear and log regression models using house prices against number of bedrooms, bathrooms and parking. Residuals are analysed and predictions can be made. Currently this is proof of concept and is severely inaccurate, underestimating true values.




## Old Scripts
**domain_scraper.py** is an early version of the later script, only looking at a single page of results from the website.


## Setup Scripts
**sql_setup** simply initialised the local sql database and table which is interacted with in part 2.
**suburb_postcode_generator** took suburbs and postcodes copied from the internet into **suburbs.txt** and created the more readable json format used in Part 1.