# House Price Estimator



## Part 1: Obtaining house price data

The first portion of this project deals with obtaining house price data from domain.com.au using the scrapfly API.

The first script is a simple example which takes a list of url strings and returns the data as a json is **domain_scraper.py** This is in the folder setupScripts, and does not need to be run.

A more complex script is **domain_scraper2.py** which searches through result pages in a given postcode for the listing data and creates a json file.
This script reuses functions from domain_scraper.py, but they are repeated in domain_scraper2.py for clarity. A scrapfly API key is required, and the script takes inputs of a base url (including suburb information), and the number of pages to be scraped. Running this multiple times will append data onto the existing json file. 


## Part 2: Creating an SQL database of the scraped data

**sql_setup** is a short script creating a database called "houses" on the local MySQL system which does not need to be run after the first time.

The database setup is performed by **sql_interpreter** which takes the json output of **domain_scraper2.py** and connects to the MySQL database to create (overwrite) the json file. 



## Part 3: Accessing the SQL database

The **price_estimator** accesses the SQL database. The desired columns can be obtained, and to find desired column names, see **SQL_TABLE_Values.sql**.
