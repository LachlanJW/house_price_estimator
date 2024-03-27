import json
import asyncio
import jmespath
from httpx import AsyncClient, Response
from parsel import Selector
from typing import List, Dict


# This code has been copied from a scrapfly.io tutorial:
# https://scrapfly.io/blog/how-to-scrape-domain-com-au-real-estate-property-data/


# Paste the urls to be searched into this list,
# they will be used in the run function at the bottom of the page
URLS = [
    "https://www.domain.com.au/19-2-archibald-street-lyneham-act-2602-2019067942", # noqa
    "https://www.domain.com.au/9-fox-place-lyneham-act-2602-2019071650" # noqa
]


client = AsyncClient(
    http2=True,
    # Add basic browser headers to mimize blocking chances
    headers={
        "accept-language": "en-US,en;q=0.9",
        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                       "AppleWebKit/537.36 (KHTML, like Gecko)"
                       "Chrome/96.0.4664.110"
                       "Safari/537.36"),
        "accept": ("text/html,"
                   "application/xhtml+xml,"
                   "application/xml;"
                   "q=0.9,"
                   "image/webp,"
                   "image/apng,"
                   "*/*;q=0.8"),
        "accept-encoding": "gzip, deflate, br",
    }
)


def parse_property_page(data: Dict) -> Dict:
    """Refine property pages data"""
    if not data:
        return {}
    result = jmespath.search(
        """{
    listingId: listingId,
    listingUrl: listingUrl,
    unitNumber: unitNumber,
    streetNumber: streetNumber,
    street: street,
    suburb: suburb,
    postcode: postcode,
    createdOn: createdOn,
    propertyType: propertyType,
    beds: beds,
    phone: phone,
    agencyName: agencyName,
    propertyDeveloperName: propertyDeveloperName,
    agencyProfileUrl: agencyProfileUrl,
    propertyDeveloperUrl: propertyDeveloperUrl,
    description: description,
    loanfinder: loanfinder,
    schools: schoolCatchment.schools,
    suburbInsights: suburbInsights,
    gallery: gallery,
    listingSummary: listingSummary,
    agents: agents,
    features: features,
    structuredFeatures: structuredFeatures,
    faqs: faqs
    }""",
        data,
    )
    return result or {}


def parse_hidden_data(response: Response) -> Dict:
    """Parse json data from script tags"""
    selector = Selector(response.text)
    script = selector.xpath("//script[@id='__NEXT_DATA__']/text()").get()

    if script is None:
        return {}  # Return an empty dictionary if script is None

    data = json.loads(script)
    return data["props"]["pageProps"]["componentProps"]


async def scrape_properties(urls: List[str]) -> List[Dict]:
    """Scrape listing data from property pages"""
    # add the property page URLs to a scraping list
    to_scrape = [client.get(url) for url in urls]
    properties = []
    # scrape all the property page concurrently
    for response_future in asyncio.as_completed(to_scrape):
        response = await response_future  # Await Future object for response
        assert response.status_code == 200, "request has been blocked"
        data = parse_hidden_data(response)
        data = parse_property_page(data)
        properties.append(data)
    print(f"scraped {len(properties)} property listings")
    return properties


async def run():
    data = await scrape_properties(URLS)

    # print the data in JSON format
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(run())
