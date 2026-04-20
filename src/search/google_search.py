import os

import requests
import json
import logging
import PyPDF2
import io

from ddgs import DDGS

logger = logging.getLogger(__name__)


def scrape_pdf_content(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))

            # Get the number of pages in the PDF
            num_pages = len(pdf_reader.pages)

            content = ''

            # Iterate over each page and extract the text
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                content += page.extract_text()

            return {'content': content}
        else:
            logging.error(f"Error downloading PDF from "
                          f"{url}: Status code {response.status_code}")
            return {'error': f"Error downloading PDF: Status code {response.status_code}"}

    except Exception as e:
        logging.error(f"Error scraping PDF content from {url}: {e}")
        return {'error': str(e)}


def scrape_content(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)\
            AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36'}
    logger.info(f"Retrieving content from {url}")
    try:
        response = requests.get(url, timeout=5, headers=headers)
    except:
        response = None
    if response is None or response.status_code != 200:
        logger.error(f"Failed to retrieve {url}")
        return {'content': '', 'error': f"Failed to retrieve {url}"}

    content = response.text
    return {'content': content}


def duckduckgo_search_engine(config, query):
    logger.info(f"Searching DuckDuckGo for: {query}")
    results = []

    # First try TAMU-scoped search for highly relevant results
    tamu_query = f"site:tamu.edu {query}"
    try:
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(tamu_query, max_results=config.NUM_SEARCH_RESULTS))
        logger.info(f"TAMU-scoped search returned {len(ddg_results)} results.")
    except Exception as e:
        logger.error(f"DuckDuckGo TAMU-scoped search failed: {e}")
        ddg_results = []

    # Fallback to general search if scoped search is insufficient
    if len(ddg_results) < 2:
        logger.info("Falling back to broader DuckDuckGo search.")
        try:
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(f"Texas A&M University TAMU {query}", max_results=config.NUM_SEARCH_RESULTS))
        except Exception as e:
            logger.error(f"DuckDuckGo broad search failed: {e}")
            return []

    if not ddg_results:
        print("No search results found.")
        return []

    for item in ddg_results:
        url = item.get('href', '')
        if not url:
            continue

        if url.lower().endswith('.pdf'):
            metadata = scrape_pdf_content(url)
        else:
            metadata = scrape_content(url)

        if 'error' in metadata:
            logger.warning(f"Scraping blocked for {url}, falling back to DDG snippet.")
            metadata = {'content': item.get('body', '')}

        result = {
            'url': url,
            'title': item.get('title', ''),
            'description': item.get('body', ''),
            'metadata': metadata
        }

        results.append(result)

    with open(os.path.join(config.BASE_PATH, "search_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.debug("Scraped results stored in search_results.json")
    return results
