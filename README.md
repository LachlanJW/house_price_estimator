# House Price Estimator

## Part 1: Obtaining house price data

The first portion of this project deals with obtaining house price data from domain.com.au using code from scrapfly:
"https://scrapfly.io/blog/how-to-scrape-domain-com-au-real-estate-property-data/"

A simple example which takes a list of url strings and returns the data as a json is **domain_scraper.py**

A more complex script is **domain_scraper2.py** which searches through result pages in a given postcode for the listing data.
This file reuses functions from domain_scraper.py, but they are repeated in domain_scraper2.py for clarity.
