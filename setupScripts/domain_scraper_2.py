# This code has been inspired by the scrapfly.io tutorial:
# https://scrapfly.io/blog/how-to-scrape-domain-com-au-real-estate-property-data/

import os
import asyncio
import json
import jmespath
from dotenv import load_dotenv
from scrapfly import ScrapeConfig, ScrapflyClient, ScrapeApiResponse  # type: ignore # noqa
from typing import Dict, List
from loguru import logger as log


# Alter the url provided to reflect the desired suburb, state and postcode
# Alter the number of pages you want scraped
URL = ("https://www.domain.com.au/sold-listings/wright-act-2611/?excludepricewithheld=1") # noqa
SCRAPE_PAGES = 50

load_dotenv()

# Make sure to set a Scrapfly API key in the environ
SCRAPFLY = ScrapflyClient(key=os.environ["SCRAPFLY_KEY"])


BASE_CONFIG = {
    # Bypass domain.com.au scraping blocking
    "asp": True,
    "country": "AU",
    "cache": True
}


def parse_hidden_data(response: ScrapeApiResponse) -> Dict:
    """Parse json data from script tags"""
    selector = response.selector
    script = selector.xpath("//script[@id='__NEXT_DATA__']/text()").get()
    data = json.loads(script)
    return data["props"]["pageProps"]["componentProps"]


def parse_property_page(data: Dict) -> Dict:
    """Refine property pages data"""
    if not data:
        return {}
    result = jmespath.search(
        """{
    listingId: listingId,
    price: price,
    street: street,
    suburb: suburb,
    state: state,
    postcode: postcode,
    propertyType: propertyType,
    lat: lat,
    lng: lng,
    beds: beds,
    baths: baths,
    parking: parking,
    landsize: landsize,
    features: features,
    tagText: tagText,
    }""",
        data,
    )
    return result


def parse_search_page(data: Dict) -> List[Dict]:
    """Refine search pages data"""
    if not data:
        return [{}]
    data = data["listingsMap"]
    result = []
    # Obtain only the required data
    for key in data.keys():
        item = data[key]
        parsed_data = jmespath.search(
            """{ id: id,
                 price: listingModel.price,
                 address: listingModel.address,
                 features: listingModel.features
                 date: listingModel.tags.tagText  }""",
            item,
        )
        result.append(parsed_data)
    return result


async def scrape_properties(urls: List[str]) -> List[Dict]:
    """Scrape listing data from property pages"""
    # Add the property page URLs to a scraping list
    to_scrape = [ScrapeConfig(url, **BASE_CONFIG) for url in urls]
    properties = []
    # Scrape all the property page concurrently
    async for response in SCRAPFLY.concurrent_scrape(to_scrape):
        # Parse the data from script tag
        data = parse_hidden_data(response)
        # Append the data to the list after refining
        properties.append(parse_property_page(data))
    log.success(f"scraped {len(properties)} property listings")
    return properties


async def scrape_search(url: str, max_scrape_pages: int) -> List[Dict]:
    """Scrape property listings from search pages"""

    first_page = await SCRAPFLY.async_scrape(ScrapeConfig(url, **BASE_CONFIG))
    log.info("scraping search page {}", url)
    data = parse_hidden_data(first_page)
    search_data = parse_search_page(data)

    # Get the number of maximum search pages
    max_search_pages = data["totalPages"]

    # Scrape the number of pages desired, or the total pages, whichever is less
    max_scrape_pages = min(max_scrape_pages, max_search_pages)

    log.info(
        f"scraping ({max_scrape_pages} pages)"
    )
    # add the remaining search pages to a scraping list
    other_pages = [
        ScrapeConfig(
            # add a "?page" parameter at the end of the URL
            str(first_page.context["url"]) + f"?page={page}",
            **BASE_CONFIG,
        )
        for page in range(2, max_scrape_pages + 1)
    ]
    # Scrape the remaining search pages concurrently
    async for response in SCRAPFLY.concurrent_scrape(other_pages):
        # parse the data from script tag        
        data = parse_hidden_data(response)
        # Append the data to the list after refining        
        search_data.extend(parse_search_page(data))
    log.success(f"scraped ({len(search_data)}) from {url}")
    return search_data


async def run():
    print("running Domain.com.au scrape")
    # Use url and page scrape values provided at top of document
    search_data = await scrape_search(url=URL, max_scrape_pages=SCRAPE_PAGES)

    # Append this output to "data.json"
    with open('data.json', 'r', encoding='utf-8') as file:
        existing_data = json.load(file)
        total_data = existing_data.extend(search_data)

    with open('data.json', 'w', encoding='utf-8') as file:
        json.dump(total_data, file, ensure_ascii=False, indent=4)

    return

if __name__ == "__main__":
    asyncio.run(run())
